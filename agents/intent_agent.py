import re
import sys
import json
import difflib
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta
from energy_agentic_ai.utils import normalize_datetime

sys.path.append('/content')

# -----------------------------------------------------------------------------------
#  This class detects intents after parsing natural language queries.
# -----------------------------------------------------------------------------------
class IntentAgent:
    def __init__(self, llm=None, region_list=None):
        self.llm = llm
        self.region_list = region_list or []
        self.region_aliases = self._build_region_aliases()
        self.actions = [
            "peak_demand",
            "all_demands",
            "total_demand",
            "average_demand",
            "outage_summary",
            "average_outage_duration",
            "anomaly_detection",
            "free_text"
        ]


    def extract_date_from_query(self, query: str):
        """
        Extracts a single date or date range from a natural language query string.
        Handles formats like:
        - 'on Jan 08, 2025'
        - 'between July 1, 2025 and July 10, 2025'
        - 'from 2025-07-01 to 2025-07-10'
        Returns (start_date, end_date) as strings in '%d-%b-%Y' format.
        """
        query = query.strip()

        # Handle single 4-digit year
        # Match a standalone 4-digit year not part of another date
        year_matches = re.findall(r"(?<![\d/-])(20\d{2})(?![\d/-])", query)
        if len(year_matches) == 1:
            year = int(year_matches[0])
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31)
            return {"start_date": start.strftime("%d-%b-%Y"), "end_date": end.strftime("%d-%b-%Y")}

        # Handle date ranges first
        range_match = re.search(
            r"(?:between|from)\s+([A-Za-z0-9,\-/ ]+?)\s+(?:and|to)\s+([A-Za-z0-9,\-/ ]+)",
            query, re.IGNORECASE
        )
        if range_match:
            try:
                start = parser.parse(range_match.group(1)).strftime("%d-%b-%Y")
                end = parser.parse(range_match.group(2)).strftime("%d-%b-%Y")
                return {"start_date": start, "end_date": end}
            except Exception:
                pass

        # Handle single date with 'on'
        single_match = re.search(
            r"\bon\s+([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4}|[A-Za-z]{3,9} [0-9]{1,2}, [0-9]{4}|[0-9]{4}-[0-9]{2}-[0-9]{2})", query, re.IGNORECASE
        )
        if single_match:
            try:
                start = parser.parse(single_match.group(1)).strftime("%d-%b-%Y")
                return {"start_date": start, "end_date": start}
            except Exception:
                pass

        today = datetime.today()
        if "last week" in query:
            start = today - timedelta(days=today.weekday() + 7)
            end = start + timedelta(days=6)
            return {"start_date": start.strftime("%d-%b-%Y"), "end_date": end.strftime("%d-%b-%Y")}
        if "this week" in query:
            start = today - timedelta(days=today.weekday())
            end = start + timedelta(days=6)
            return {"start_date": start.strftime("%d-%b-%Y"), "end_date": end.strftime("%d-%b-%Y")}
        if "last month" in query:
            first_this_month = today.replace(day=1)
            last_month_end = first_this_month - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            return {"start_date": last_month_start.strftime("%d-%b-%Y"), "end_date": last_month_end.strftime("%d-%b-%Y")}
        if "this month" in query:
            start = today.replace(day=1)
            end = today
            return {"start_date": start.strftime("%d-%b-%Y"), "end_date": end.strftime("%d-%b-%Y")}
        if "yesterday" in query:
            y = today - timedelta(days=1)
            return {"start_date": y.strftime("%d-%b-%Y"), "end_date": y.strftime("%d-%b-%Y")}
        if "today" in query:
            return {"start_date": today.strftime("%d-%b-%Y"), "end_date": today.strftime("%d-%b-%Y")}
        return {"start_date": None, "end_date": None}

    def _build_region_aliases(self):
        aliases = {}
        for region in self.region_list:
            code = region
            lower = region.lower()
            for variant in [lower, code] + lower.split():
                aliases[variant] = region
        return aliases

    def detect_region(self, query: str):
        q = query.lower()
        for alias, code in self.region_aliases.items():
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return code
        return None

    def detect_year(self, query: str):
        match = re.search(r"\b(20\d{2}|19\d{2})\b", query)
        if match:
            return int(match.group(0))
        return None

    def _rule_based_intent(self, query: str) -> dict:
        q = query.lower()
        region = self.detect_region(query)
        year = self.detect_year(query)
        time_window = self.extract_date_from_query(query)

        # --- Demand actions ---
        if any(k in q for k in ["peak", "highest", "max demand", "top load"]):
            action = "peak_demand"
        elif any(k in q for k in ["total demand", "sum of demand", "aggregate demand", "overall demand"]):
            action = "total_demand"
        elif any(k in q for k in ["average demand", "mean demand", "typical demand"]):
            action = "average_demand"
        elif any(k in q for k in ["demand", "demands"]):
            action = "all_demands"

        # --- Outage actions ---
        elif any(k in q for k in ["average outage", "mean outage duration", "typical outage"]):
            action = "average_outage_duration"
        elif any(k in q for k in ["outage", "blackout", "power cut", "failure report"]):
               # detect structured intent
              if "by region" in q or "by area" in q or "count" in q or "total" in q or "hours" in q or "how many" in q or "duration" in q:
                  action = "structured_outage_summary"
              else:
                  action = "outage_summary"

        # --- Anomalies ---
        elif any(k in q for k in ["anomaly", "abnormal", "irregular", "unusual"]):
            action = "anomaly_detection"
        else:
            action = "free_text"

        return {
            "action": action,
            "region": region,
            "year": year,
            "start_date": time_window["start_date"],
            "end_date": time_window["end_date"],
        }

    def _llm_intent(self, query: str) -> dict:
        """Ask the LLM to classify query into structured JSON"""
        if not self.llm:
            return {"action": "free_text", "query": query}

        prompt = f"""
        You are an intent classifier for energy data analytics.
        Classify the query below into JSON with keys:
        - action: one of ["peak_demand", "outage_summary", "anomaly_detection", "free_text"]
        - region: optional
        - year: optional
        - metric: optional
        Query: "{query}"
        Respond only with JSON.
        """

        try:
            response = self.llm(prompt, max_length=128, do_sample=False)[0]["generated_text"]
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if parsed.get("action") not in self.actions:
                    parsed["action"] = difflib.get_close_matches(parsed.get("action", ""), self.actions, n=1, cutoff=0.4)[0] if difflib.get_close_matches(parsed.get("action", ""), self.actions, n=1, cutoff=0.4) else "free_text"
                return parsed
        except Exception:
            pass
        return {"action": "free_text", "query": query}

    def parse(self, query: str) -> dict:
        if not query or not query.strip():
            return {"action": "free_text", "query": ""}
        rule_result = self._rule_based_intent(query)
        if rule_result:
            return rule_result
        llm_result = self._llm_intent(query)
        return llm_result    

