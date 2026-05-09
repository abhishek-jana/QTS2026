import os
import yaml
import sys
from datetime import datetime
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.getcwd())

from qts_core.logger import logger
from research_lab.data_engine import DataEngine
from research_lab.real_data_ingestor import InstitutionalIngestor

def run_historical_ingestion():
    load_dotenv()
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    db_path = config['data_engine']['storage_path']
    # Specific missing tickers
    tickers = ["HON", "MS", "CVS", "COP", "IBM", "BA", "SPGI", "CAT", "LMT", "RTX"]
    
    # Initialize Engine & Ingestor
    engine = DataEngine(storage_path=db_path)
    ingestor = InstitutionalIngestor(data_engine=engine, config=config)
    
    # Range: 2016-01-01 to 2020-07-27
    start_str = "2016-01-01"
    end_str = "2020-07-27"
    
    logger.info(f"🚀 Recovering missing Historical Ingestion for {len(tickers)} tickers...")
    
    try:
        ingestor.ingest_universe(tickers, start_str, end_str)
        logger.success("✅ Missing Tickers Recovered!")
    except Exception as e:
        logger.error(f"❌ Recovery Failed: {e}")

if __name__ == "__main__":
    run_historical_ingestion()
