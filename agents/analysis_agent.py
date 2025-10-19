import re
import sys
import duckdb
import pandas as pd
from datetime import datetime
from dateutil import parser
from energy_agentic_ai.utils import normalize_datetime

sys.path.append('/content')

class AnalysisAgent:
    def __init__(self, df_consumption, df_outages=None):
        self.df_consumption = df_consumption
        self.df_outages = df_outages
        self.con = duckdb.connect()
        self.con.register('df_consumption_var', self.df_consumption)
        if self.df_outages is not None:
            self.con.register('df_outages_var', self.df_outages)

    # -------------------------------
    # Demand Queries
    # -------------------------------
    def get_all_demands(self, region=None, start_date=None, end_date=None):
        query = "SELECT Date, Region, Demand_MW as Demand FROM df_consumption_var"
        conditions = []
        if region:
            conditions.append(f"Region = '{region}'")
        if start_date and end_date:
            conditions.append(
                f"STRPTIME(Date, '%d-%b-%Y') BETWEEN STRPTIME('{start_date}', '%d-%b-%Y') "
                f"AND STRPTIME('{end_date}', '%d-%b-%Y')"
            )
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY STRPTIME(Date, '%d-%b-%Y') ASC"

        result = self.con.execute(query).fetchdf()
        if result.empty:
            return None

        result["Date"] = result["Date"].apply(normalize_datetime)
        return result.to_dict(orient="records")

    def get_peak_demand(self, region=None, start_date=None, end_date=None):
        query = "SELECT Date, Region, Demand_MW as PeakDemand FROM df_consumption_var"
        conditions = []
        if region:
            conditions.append(f"Region = '{region}'")
        if start_date and end_date:
            conditions.append(
                f"STRPTIME(Date, '%d-%b-%Y') BETWEEN STRPTIME('{start_date}', '%d-%b-%Y') "
                f"AND STRPTIME('{end_date}', '%d-%b-%Y')"
            )
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY Demand_MW DESC LIMIT 1"

        result = self.con.execute(query).fetchdf()
        if result.empty:
            return None
        row = result.iloc[0]
        return {
            "Date": normalize_datetime(row["Date"]),
            "Region": row["Region"],
            "PeakDemand": row["PeakDemand"]
        }

    def get_total_demand(self, region=None, start_date=None, end_date=None):
        """Compute total demand (SUM)."""
        query = "SELECT Region, SUM(Demand_MW) AS TotalDemand FROM df_consumption_var"
        conditions = []
        if region:
            conditions.append(f"Region = '{region}'")
        if start_date and end_date:
            conditions.append(
                f"STRPTIME(Date, '%d-%b-%Y') BETWEEN STRPTIME('{start_date}', '%d-%b-%Y') "
                f"AND STRPTIME('{end_date}', '%d-%b-%Y')"
            )
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " GROUP BY Region"

        df = self.con.execute(query).fetchdf()
        if df.empty:
            return None
        return df.to_dict(orient="records")

    def get_average_demand(self, region=None, start_date=None, end_date=None):
        """Compute average demand (AVG)."""
        query = "SELECT Region, AVG(Demand_MW) AS AverageDemand FROM df_consumption_var"
        conditions = []
        if region:
            conditions.append(f"Region = '{region}'")
        if start_date and end_date:
            conditions.append(
                f"STRPTIME(Date, '%d-%b-%Y') BETWEEN STRPTIME('{start_date}', '%d-%b-%Y') "
                f"AND STRPTIME('{end_date}', '%d-%b-%Y')"
            )
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " GROUP BY Region"

        df = self.con.execute(query).fetchdf()
        if df.empty:
            return None
        return df.to_dict(orient="records")

    def get_regional_peak_summary(self):
        """Returns a DataFrame with Region, PeakDemand, and Date of peak."""
        query = """
            SELECT Region, Demand_MW as PeakDemand, Date
            FROM df_consumption_var AS t1
            WHERE Demand_MW = (
                SELECT MAX(Demand_MW)
                FROM df_consumption_var AS t2
                WHERE t2.Region = t1.Region
            )
            ORDER BY Region
        """
        df = self.con.execute(query).fetchdf()
        df["Date"] = df["Date"].apply(normalize_datetime)
        return df

    # -------------------------------
    # Outage Queries
    # -------------------------------
    def extract_duration(self, text: str) -> float:
        text = text.lower()
        hyphen_hour_match = re.search(r'(\d+(?:\.\d+)?)-hour', text)
        if hyphen_hour_match:
            return float(hyphen_hour_match.group(1))
        hour_match = re.search(r'(\d+(?:\.\d+)?)\s*hour', text)
        if hour_match:
            return float(hour_match.group(1))
        min_match = re.search(r'(\d+(?:\.\d+)?)\s*(minute|min)', text)
        if min_match:
            return float(min_match.group(1)) / 60
        if "complete loss" in text or "system separation" in text:
            return 2.0
        if "unexpected transmission" in text:
            return 1.0
        if "physical threat" in text or "cyber event" in text:
            return 0.5
        return 0.0


    def summarize_outages_by_region(self, region=None, year=None, start_date=None, end_date=None):
        if self.df_outages is None or self.df_outages.empty:
            return pd.DataFrame(columns=["Region", "TotalOutages", "TotalHours"])

        df = self.df_outages.copy()
        df["Duration_hr"] = df["Report_Text"].apply(self.extract_duration)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

        query = """
            SELECT Region,
                  COUNT(*) AS TotalOutages,
                  SUM(Duration_hr) AS TotalHours
            FROM df
        """

        conditions = []
        if region:
            conditions.append(f"Region = '{region}'")
        if start_date and end_date:
            conditions.append(
                f"Date BETWEEN STRPTIME('{start_date}', '%d-%b-%Y') AND STRPTIME('{end_date}', '%d-%b-%Y')"
            )

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY Region ORDER BY Region"

        result = duckdb.query_df(df, "df", query).to_df()
        return result

    def get_average_outage_duration(self, region=None, year=None):
        if self.df_outages is None or self.df_outages.empty:
            return pd.DataFrame(columns=["Region", "AverageOutageDuration"])
        df = self.df_outages.copy()
        df["Duration_hr"] = df["Report_Text"].apply(self.extract_duration)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        if region:
            df = df[df["Region"] == region]
        if year:
            df = df[df["Date"].dt.year == year]
        if df.empty:
            return pd.DataFrame(columns=["Region", "AverageOutageDuration"])
        query = """
            SELECT Region, AVG(Duration_hr) AS AverageOutageDuration
            FROM df
            GROUP BY Region
            ORDER BY Region
        """
        result = duckdb.query_df(df, "df", query).to_df()
        return result

    # -------------------------------
    # Anomaly Detection Placeholder
    # -------------------------------
    def run_anomaly_detection(self):
        return [{"Date": "2025-01-01", "Region": "North", "Issue": "Demand spike"}]
