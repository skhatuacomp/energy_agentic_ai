import os
import sys
import pandas as pd
from datetime import datetime
from dateutil import parser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from energy_agentic_ai.utils import normalize_datetime

sys.path.append('/content')

# -----------------------------------------------------------------------------------
#  This class loads consumption and outages log csv files. 
#  It also embed outages log into vector db.
# -----------------------------------------------------------------------------------
class DataAgent:
    def __init__(self, consumption_file=None, outage_file=None):
        self.data_dir = "/content/energy_agentic_ai/data"
        self.consumption_df = None
        self.outage_df = None
        self.vectorstore = None
        self.load_data(consumption_file, outage_file)

    def load_data(self, consumption_file=None, outage_file=None):
        """
        Load data from uploaded files (Streamlit UI) or from default CSV paths.
        """
        try:
            # Load consumption data
            if consumption_file is not None:
                consumption_file.seek(0)
                self.consumption_df = pd.read_csv(consumption_file)
                self.consumption_df['Date'] = self.consumption_df['Date'].apply(normalize_datetime)
                print("✅ Loaded consumption data from uploaded file.")
            else:
                default_consumption_path = os.path.join(self.data_dir, "consumption.csv")
                if os.path.exists(default_consumption_path):
                    self.consumption_df = pd.read_csv(default_consumption_path)
                    self.consumption_df['Date'] = self.consumption_df['Date'].apply(normalize_datetime)
                    print("✅ Loaded consumption data from default path.")
                else:
                    raise FileNotFoundError("Consumption data file not found.")
            # Load outage data
            if outage_file is not None:
                outage_file.seek(0)
                self.outage_df = pd.read_csv(outage_file)
                self.outage_df['Date'] = self.outage_df['Date'].apply(normalize_datetime)
                print("✅ Loaded outage data from uploaded file.")
            else:
                default_outage_path = os.path.join(self.data_dir, "outages.csv")
                if os.path.exists(default_outage_path):
                    self.outage_df = pd.read_csv(default_outage_path)
                    self.outage_df['Date'] = self.outage_df['Date'].apply(normalize_datetime)
                    print("✅ Loaded outage data from default path.")
                else:
                    raise FileNotFoundError("Outage data file not found.")
            self.embed_outage_reports()
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def embed_outage_reports(self, persist_dir="chroma_db"):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        texts = [
            f"Date: {d}, Region: {r}, Report: {t}"
            for d, r, t in zip(self.outage_df["Date"], self.outage_df["Region"], self.outage_df["Report_Text"])
        ]

        metadatas = [
            {"Date": str(d), "Region": r}
            for d, r in zip(self.outage_df["Date"], self.outage_df["Region"])
        ]

        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            persist_directory=persist_dir
        )
        print("✅ Outage reports embedded and stored in ChromaDB.")
