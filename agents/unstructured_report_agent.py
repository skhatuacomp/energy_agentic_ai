import os
import re
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

# -----------------------------------------------------------------------------------
#  This class generates reports for unstructured analyses date.
# -----------------------------------------------------------------------------------
class UnstructuredReportAgent:

    def __init__(self, chroma_path="/content/chroma_db", model_id="HuggingFaceH4/zephyr-7b-beta", top_k=10):
        """
        Initialize the agent with:
        - Local ChromaDB store for embedded outage reports
        - Hugging Face Zephyr model for summarization
        """
        self.chroma_path = chroma_path
        self.model_id = model_id
        self.top_k = top_k

        # Initialize ChromaDB embeddings & vectorstore
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=embedding_model)
        docs = self.db.get()

        self.retriever = self.db.as_retriever(search_kwargs={"k": self.top_k})

        # Initialize Hugging Face Inference API client
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("❌ Missing Hugging Face API token. Set HUGGINGFACEHUB_API_TOKEN in environment.")
        self.client = InferenceClient(model=self.model_id, token=hf_token)

    def query_outage_reports(self, query: str, region=None, start_date=None, end_date=None) -> str:
        """
        Retrieve relevant outage reports from ChromaDB and summarize concisely using Zephyr.
        Ensures minimal, factual output.
        """
        try:
            docs = self.retriever.invoke(query)

            # Deduplicate by metadata + content
            unique_docs = {}
            for doc in docs:
                key = (doc.metadata.get("Date"), doc.metadata.get("Region"), doc.page_content)
                unique_docs[key] = doc
            docs = list(unique_docs.values())

            # Apply date filtering if provided
            if start_date and end_date:
                try:
                    start_dt = datetime.strptime(start_date, "%d-%b-%Y")
                    end_dt = datetime.strptime(end_date, "%d-%b-%Y")
                except ValueError:
                    return "⚠️ Invalid date format. Use DD-MMM-YYYY (e.g., 08-Jan-2025)."

                filtered_docs = [
                    doc for doc in docs
                    if "Date" in doc.metadata and
                    start_dt <= datetime.strptime(doc.metadata["Date"], "%d-%b-%Y") <= end_dt
                ]
            else:
                filtered_docs = docs

            if not filtered_docs:
                return "No relevant outage reports found."

            # Combine context but keep it internal
            combined_text = "\n".join([doc.page_content for doc in filtered_docs])

            # Very strict system prompt for short factual answers
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise data summarizer for power outage reports. "
                        "Answer in one short factual sentence using only the known data. "
                        "Do not include speculation, reasoning, or follow-up questions. "
                        "Do not restate the query or mention unavailable data. "
                        "Output should be under 25 words, strictly factual."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{query}\n\n[The agent has access to outage data internally.]",
                },
            ]

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=60,
                temperature=0.0,  # fully deterministic and concise
            )

            if not response.choices or not response.choices[0].message:
                return "⚠️ No output from Zephyr model."

            text = response.choices[0].message["content"].strip()
            clean_text = re.sub(r"\[/?(INST|USER|ASS)\]", "", text).strip()
            return clean_text

        except Exception as e:
            return f"⚠️ Error during inference: {str(e)}"

