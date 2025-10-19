import os
import importlib
import sys
from getpass import getpass

# Safe reload for each agent file
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

from energy_agentic_ai.agents.analysis_agent import AnalysisAgent
from energy_agentic_ai.agents.data_agent import DataAgent
from energy_agentic_ai.agents.structured_report_agent import StructuredReportAgent
from energy_agentic_ai.agents.unstructured_report_agent import UnstructuredReportAgent
from energy_agentic_ai.agents.intent_agent import IntentAgent

sys.path.append('/content')

def main():
    print("🚀 Starting Agentic Energy Management Assistant")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<hf_your_token_here>"

    # Step 1: Load and embed data
    data_agent = DataAgent()
    docs = data_agent.vectorstore.get()

    # Step 2: Generate analyses agent
    analysis_agent = AnalysisAgent(data_agent.consumption_df,data_agent.outage_df)

    # Step 3: Generate report agent
    structured_report_agent = StructuredReportAgent()
    unstructured_report_agent = UnstructuredReportAgent(chroma_path = "/content/chroma_db")

    # Step 4: Generate intent agent
    region_list = sorted(data_agent.consumption_df['Region'].unique().tolist())
    intent_agent = IntentAgent(region_list=region_list)

    # Step 5: Execute User Query
    query = "How long did the power outage last in Region South on 08th Feb, 2025?"
    print("\n🧠 Query:", query)
    # Parse intent
    intent = intent_agent.parse(query)
    action = intent.get("action")
    region = intent.get("region")
    year = intent.get("year")
    start_date = intent.get("start_date")
    end_date = intent.get("end_date")

    # ---------------------------
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

    elif action == "anomaly_detection":
        result_data = analysis_agent.run_anomaly_detection()
        result = structured_report_agent.simple_text_report(result_data)

    elif action == "structured_outage_summary":
        result_data = analysis_agent.summarize_outages_by_region(region, year, start_date, end_date)
        result = structured_report_agent.generate_outage_summary(result_data, query)

    elif action == "free_text" or action == "outage_summary":
        result = unstructured_report_agent.query_outage_reports(query, region, start_date, end_date)

    else:
        result = "Sorry, I could not understand your query. Try asking about 'peak demand', 'total demand', 'average demand', 'outage summary', or 'average outage duration'."


    print("💬", result)


if __name__ == "__main__":
    main()
