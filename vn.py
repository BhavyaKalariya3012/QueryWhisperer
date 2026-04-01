import os
import sqlite3
from dotenv import load_dotenv, find_dotenv
import warnings
from typing import Dict, Optional, Annotated, Tuple, List
from pathlib import Path
import yaml
from openai import OpenAI as OpenAIClient
import yfinance as yf

from vanna.openai import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from tqdm import tqdm

_ = load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

__curdir__ = os.getcwd()
if "notebooks" in __curdir__:
    chroma_path = "../chroma_store"
    yaml_file_path = "../config/training.yaml"
else:
    chroma_path = "./chroma_store"
    yaml_file_path = "./config/training.yaml"

db_path = "./database/stocks.db"

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """Our main Vanna handler. Inherits from Vanna's ChromaDB_VectorStore
    and OpenAI_Chat which implement many of the abstract methods in Vanna's
    Base class. The only thing we need to do is define the constructor where
    we initialize the different objects."""
    
    def __init__(
        self, 
        config: Optional[Annotated[
            Dict[str, str], 
            """The configuration for the vector store or the LLM you want to use.
            For e.g. {'model': 'llama3-8b-8192'}"""
        ]]
    ) -> None:
        config = config or {}
        groq_api_key = config.get("api_key") or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your environment before starting the app.")

        client = OpenAIClient(
            api_key=groq_api_key,
            base_url=config.get("api_base", os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")),
        )

        llm_config = {k: v for k, v in config.items() if k != "api_base"}

        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=llm_config)

def load_query_data(yaml_file_path: str) -> List[Tuple[str, str]]:
    """Returns a list of training queries for LLM training.
    
    This is a mission critical function.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            try:
                documents = yaml.safe_load(file)
                if not isinstance(documents, list):
                    raise ValueError("YAML content is not a list of documents.")
            except yaml.YAMLError as yaml_error:
                raise ValueError(f"Error parsing YAML file: {yaml_error}") from yaml_error
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"Error: The file at {yaml_file_path} was not found.") from fnf_error
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}") from e

    return [(doc.get('question'), doc.get('answer')) for doc in documents]


def ensure_sqlite_database(db_file_path: str) -> None:
    """Create the demo SQLite database on first run if it does not exist."""
    if os.path.exists(db_file_path):
        return

    print(f"Database not found at {db_file_path}. Bootstrapping from Yahoo Finance...")
    Path(db_file_path).parent.mkdir(parents=True, exist_ok=True)

    ticker_map = {
        "ilmn": "ILMN",
        "aapl": "AAPL",
        "nvda": "NVDA",
    }

    with sqlite3.connect(db_file_path) as connection:
        for table_name, ticker in ticker_map.items():
            df = yf.download(ticker, period="10y", progress=False, auto_adjust=True)
            if df.empty:
                raise RuntimeError(f"No market data returned for ticker {ticker}.")

            df = df.reset_index()
            # Flatten MultiIndex columns returned by newer yfinance versions
            if isinstance(df.columns, __import__('pandas').MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
            required_columns = ["date", "open", "high", "low", "close", "volume"]
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise RuntimeError(f"Missing expected columns for ticker {ticker}: {missing}")

            df = df[required_columns]
            df["date"] = df["date"].astype(str)
            df.to_sql(table_name, connection, if_exists="replace", index_label="id")


def should_train(vanna_instance: MyVanna) -> bool:
    """Train only when vector store is empty to avoid repeated startup cost."""
    try:
        training_data = vanna_instance.get_training_data()
        return training_data is None or len(training_data) == 0
    except Exception:
        return True
    
print("Instantiating Vanna...")
vn = MyVanna(config={
    "model": "llama3-8b-8192",
    "path": chroma_path, #this is the specific key that Vanna looks for (reference: Vanna's ChromaDB_VectorStore source code)
    "api_key": os.getenv("GROQ_API_KEY"),
})

print("Connecting Vanna to SQL database...")
ensure_sqlite_database(db_path)
vn.connect_to_sqlite(db_path)

if should_train(vn):
    print("Training Vanna...")
    df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
    for ddl in df_ddl['sql'].to_list():
        vn.train(ddl=ddl)
    vn.train(documentation="Illumina is defined as ilmn")
    vn.train(documentation="Apple is defined as aapl")
    vn.train(documentation="NVIDIA is defined as nvda")

    queries = load_query_data(yaml_file_path=yaml_file_path)
    for question, sql in tqdm(queries):
        vn.train(question=question, sql=sql)
else:
    print("Vanna already trained. Skipping training.")

if __name__ == "__main__":
    from vanna.flask import VannaFlaskApp
    
    app = VannaFlaskApp(vn)
    app.run()
