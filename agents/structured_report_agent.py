import os
import sys
import pandas as pd
from huggingface_hub import InferenceClient
from datetime import datetime
from energy_agentic_ai.utils import normalize_datetime

sys.path.append('/content')

# -----------------------------------------------------------------------------------
#  This class generates reports for structured analyses date.
# -----------------------------------------------------------------------------------
class StructuredReportAgent:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_name = model_name
        self.client = InferenceClient(
            model=model_name,
            token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
        )

    # -------------------------------
    # Generate demand report
    # -------------------------------
    def generate_report(self, analysis_result, query):
        if isinstance(analysis_result, pd.DataFrame):
            records = analysis_result.to_dict(orient="records")
        elif hasattr(analysis_result, "to_dict"):
            records = [analysis_result.to_dict()]
        elif isinstance(analysis_result, list):
            records = analysis_result
        else:
            records = [analysis_result]

        if not records or records[0] is None:
            return "No record found!"

        # --- Determine metric ---
        metric_key = None
        if "PeakDemand" in records[0]:
            metric_key = "PeakDemand"
            label = "Peak demand"
        elif "TotalDemand" in records[0]:
            metric_key = "TotalDemand"
            label = "Total demand"
        elif "AverageDemand" in records[0]:
            metric_key = "AverageDemand"
            label = "Average demand"
        elif "Demand" in records[0]:
            metric_key = "Demand"
            label = "Demand"
        else:
            return "Unknown demand metric in results."

        # --- Build report text ---
        sentences = []
        for r in records:
            region = r.get("Region", "N/A")
            value = r.get(metric_key, "N/A")

            date = r.get("Date")
            if date:
                date_obj = normalize_datetime(date)
                '''
                if date_obj:
                    date_str = date_obj.strftime("%Y-%m-%d")
                else:
                    date_str = str(date)
                '''
                date_str = datetime.strptime(date_obj, "%d-%b-%Y").strftime("%Y-%m-%d")
            else:
                date_str = "N/A"

            if date_str != "N/A":
                sentences.append(f"{label} observed on {date_str} in {region} with {value} MW.")
            else:
                sentences.append(f"{label} in {region} was {value} MW.")

        return "\n".join(sentences)

    # -------------------------------
    # Generate outage report
    # -------------------------------
    def generate_outage_summary(self, analysis_result, query):
        """
        Generate an outage summary report from the result of the analysis agent.
        Supports:
          - Summaries by region (total outages and hours)
          - Average outage durations
          - Totals across all regions
        Example output:
          "North: 2 outages, 5 hrs. South: 1 outage, 1 hr."
          or
          "North: Avg outage 1.5 hrs. South: Avg outage 2 hrs."
        """
        import pandas as pd

        if analysis_result is None:
            return "No outage records found!"

        # Convert possible dataframe to list of dicts
        if isinstance(analysis_result, pd.DataFrame):
            records = analysis_result.to_dict(orient="records")
        elif isinstance(analysis_result, list):
            records = analysis_result
        elif hasattr(analysis_result, "to_dict"):
            records = [analysis_result.to_dict()]
        else:
            records = [analysis_result]

        if not records:
            return "No outage data available."

        summary_parts = []

        # Auto-detect the type of outage data
        keys = records[0].keys()

        if {"TotalOutages", "TotalHours"} <= keys:
            # Region-wise totals
            for r in records:
                region = r.get("Region", "Unknown")
                outages = int(r.get("TotalOutages", 0))
                hours = r.get("TotalHours", 0)
                hours_display = int(hours) if float(hours).is_integer() else round(hours, 1)
                outage_word = "outage" if outages == 1 else "outages"
                hour_word = "hr" if hours_display == 1 else "hrs"
                summary_parts.append(f"{region}: {outages} {outage_word}, {hours_display} {hour_word}.")

        elif {"Region", "AverageOutageDuration"} <= keys:
            # Region-wise averages
            for r in records:
                region = r.get("Region", "Unknown")
                avg_dur = r.get("AverageOutageDuration", 0)
                dur_display = int(avg_dur) if float(avg_dur).is_integer() else round(avg_dur, 1)
                hour_word = "hr" if dur_display == 1 else "hrs"
                summary_parts.append(f"{region}: Avg outage {dur_display} {hour_word}.")

        elif {"TotalOutages", "TotalHours"} == keys:
            # Overall total summary
            total_outages = int(records[0].get("TotalOutages", 0))
            total_hours = records[0].get("TotalHours", 0)
            hours_display = int(total_hours) if float(total_hours).is_integer() else round(total_hours, 1)
            summary_parts.append(f"Total: {total_outages} outages, {hours_display} hrs.")

        else:
            # Fallback case
            for r in records:
                region = r.get("Region", "Unknown")
                summary_parts.append(f"{region}: Unknown outage data.")

        return " ".join(summary_parts)

