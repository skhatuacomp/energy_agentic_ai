import streamlit as st
import pandas as pd
import os
import sys
import re
from datetime import datetime
import importlib

sys.path.append('/content')

# Safe reload and import each agent file
import energy_agentic_ai.agents.data_agent as data_agent
importlib.reload(data_agent)
import energy_agentic_ai.agents.analysis_agent as analysis_agent
importlib.reload(analysis_agent)
import energy_agentic_ai.agents.structured_report_agent as structured_report_agent
importlib.reload(structured_report_agent)
import energy_agentic_ai.agents.unstructured_report_agent as unstructured_report_agent
importlib.reload(unstructured_report_agent)
import energy_agentic_ai.agents.intent_agent as intent_agent
importlib.reload(intent_agent)

from energy_agentic_ai.agents.data_agent import DataAgent
from energy_agentic_ai.agents.analysis_agent import AnalysisAgent
from energy_agentic_ai.agents.structured_report_agent import StructuredReportAgent
from energy_agentic_ai.agents.unstructured_report_agent import UnstructuredReportAgent
from energy_agentic_ai.agents.intent_agent import IntentAgent

# Set your HUGGINGFACEHUB_API_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<hf_your_token_here>"

st.title("⚡Energy Management Assistant")
# -------------------------------
# Upload CSV files from user
# -------------------------------
st.markdown(
        """
        <style>
            .stFileUploader > section {
                padding: 0;
            }
            .stFileUploader > section > div {
                display: none;
            }
            .stFileUploader > section > input + div {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
col1, col2 = st.columns(2)
with col1:
    consumption_file = st.file_uploader("Upload Consumption Data (CSV)", type=["csv"])
    st.session_state.data_agent = None
with col2:
    outage_file = st.file_uploader("Upload Outage Data (CSV)", type=["csv"])
    st.session_state.data_agent = None

# -------------------------------
# Initialize session state
# -------------------------------
for key in ["data_agent", "analysis_agent", "structured_report_agent", "unstructured_report_agent", "intent_parser",
            "peak_demand", "outage_summary", "query"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# Initialize agents
# -------------------------------
def initialize_agents():
    if st.session_state.data_agent is None or st.session_state.analysis_agent is None or st.session_state.structured_report_agent is None or st.session_state.unstructured_report_agent is None or st.session_state.intent_agent is None:
        try:
            # ---------------------------
            # Load data (from upload or from data folder)
            # ---------------------------
            if consumption_file is not None:
                consumption_df = pd.read_csv(consumption_file)
            else:
                default_path = "/content/energy_agentic_ai/data/consumption.csv"
                if os.path.exists(default_path):
                    consumption_df = pd.read_csv(default_path)
                else:
                    st.error("❌ Please upload a consumption data CSV.")
                    return False

            if outage_file is not None:
                outage_df = pd.read_csv(outage_file)
            else:
                default_path = "/content/energy_agentic_ai/data/outages.csv"
                if os.path.exists(default_path):
                    outage_df = pd.read_csv(default_path)
                else:
                    st.error("❌ Please upload an outage data CSV.")
                    return False

            data_agent = DataAgent(consumption_file, outage_file)

            if data_agent.consumption_df is None or data_agent.outage_df is None:
                st.error("❌ Could not load data. Place CSVs in energy_agentic_ai/data/.")
                return False

            analysis_agent = AnalysisAgent(data_agent.consumption_df, data_agent.outage_df)
            structured_report_agent = StructuredReportAgent()
            unstructured_report_agent = UnstructuredReportAgent(chroma_path = "/content/chroma_db")

            # Region map for intent parser
            region_list = sorted(consumption_df['Region'].unique().tolist())
            intent_agent = IntentAgent(region_list=region_list)

            st.session_state.update({
                "data_agent": data_agent,
                "analysis_agent": analysis_agent,
                "structured_report_agent": structured_report_agent,
                "unstructured_report_agent": unstructured_report_agent,
                "intent_agent": intent_agent
            })
            return True

        except Exception as e:
            st.error(f"⚠️ Initialization Error: {str(e)}")
            return False
    return True

initialize_agents()

# -------------------------------
# User input
# -------------------------------
st.markdown("""
<style>
div[data-testid="stForm"] {border: none; padding: 0; background-color: transparent;}
div[data-testid="stTextInputRootElement"] input {
    border-radius: 10px;
    border: 1px solid #ccc;
    padding: 8px;
}
div[data-testid="stFormSubmitButton"] button {
    height: 40px;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

with st.form(key="query_form", clear_on_submit=False):
    margin, col1, col2 = st.columns([0.3, 4, 1])
    with col1:
        query = st.text_input("", placeholder="Type your query here")

    with col2:
        st.write("")  # adds a small vertical spacer
        st.write("")  # you can add more for fine-tuning
        send = st.form_submit_button("➡️")

# -------------------------------
# Handle query
# -------------------------------
def extract_year(query: str):
    match = re.search(r"\b(20\d{2})\b", query)
    return int(match.group(1)) if match else None

if send and query.strip() != "":
    data_agent = st.session_state.data_agent
    analysis_agent = st.session_state.analysis_agent
    structured_report_agent = st.session_state.structured_report_agent
    unstructured_report_agent = st.session_state.unstructured_report_agent
    intent_agent = st.session_state.intent_agent

    # Parse intent
    intent = intent_agent.parse(query)
    action = intent.get("action")
    region = intent.get("region")
    year = intent.get("year")
    start_date = intent.get("start_date")
    end_date = intent.get("end_date")

    # Route to appropriate AnalysisAgent method
    # ---------------------------
    if action == "peak_demand":
        query_lower = query.lower()
        if "each region" in query_lower or "by region" in query_lower or "all regions" in query_lower:
            result_data = analysis_agent.get_regional_peak_summary()
        else:
            result_data = analysis_agent.get_peak_demand(region, start_date, end_date)
        result = structured_report_agent.generate_report(result_data, query)

    elif action == "all_demands":
        result_data = analysis_agent.get_all_demands(region, start_date, end_date)
        result = structured_report_agent.generate_report(result_data, query)

    elif action == "total_demand":
        result_data = analysis_agent.get_total_demand(region, start_date, end_date)
        result = structured_report_agent.generate_report(result_data, query)

    elif action == "average_demand":
        result_data = analysis_agent.get_average_demand(region, start_date, end_date)
        result = structured_report_agent.generate_report(result_data, query)

    elif action == "average_outage_duration":
        if region:
            result_data = analysis_agent.get_average_outage_duration(region, year)
        else:
            result_data = analysis_agent.get_average_outage_duration(None, year)
        result = structured_report_agent.generate_outage_summary(result_data, query)

    elif action == "structured_outage_summary":
        result_data = analysis_agent.summarize_outages_by_region(region, year, start_date, end_date)
        result = structured_report_agent.generate_outage_summary(result_data, query)

    elif action == "anomaly_detection":
        result_data = analysis_agent.run_anomaly_detection()
        result = structured_report_agent.simple_text_report(result_data)

    elif action == "free_text" or action == "outage_summary":
        # Route to unstructured LLM retrieval agent
        result = unstructured_report_agent.query_outage_reports(query, region, start_date, end_date)
    else:
        result = "Sorry, I could not understand your query. Try asking about 'peak demand', 'total demand', 'average demand', 'outage summary', or 'average outage duration'."

    margin, col1, col2 = st.columns([0.3, 0.2, 6])
    with col1:
        st.write("💬")
    with col2:
        st.write(result)

