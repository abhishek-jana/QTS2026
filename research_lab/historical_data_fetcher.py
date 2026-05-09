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
    tickers = config['universe']['tickers']
    
    # Initialize Engine & Ingestor
    engine = DataEngine(storage_path=db_path)
    ingestor = InstitutionalIngestor(data_engine=engine, config=config)
    
    # Target Range: 2016-01-01 to 2020-07-27 (where the current data starts)
    start_str = "2016-01-01"
    end_str = "2020-07-27"
    
    logger.info(f"🚀 Starting Historical Ingestion for {len(tickers)} tickers...")
    logger.info(f"Range: {start_str} to {end_str}")
    
    try:
        ingestor.ingest_universe(tickers, start_str, end_str)
        logger.success("✅ Historical Ingestion Complete!")
    except Exception as e:
        logger.error(f"❌ Ingestion Failed: {e}")

if __name__ == "__main__":
    run_historical_ingestion()
