import torch
import numpy as np
import pandas as pd
import yaml
import os
import sys
import duckdb
from datetime import datetime, timedelta
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.getcwd())

from qts_core.logger import logger
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.data_engine import DataEngine

def precompute_rl_data():
    logger.info("🚀 Pre-computing High-Fidelity RL Training Data...")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    db_path = config['data_engine']['storage_path']
    tickers = config['universe']['tickers']
    
    # 1. Load Market Data once
    disk_conn = duckdb.connect(db_path, read_only=True)
    logger.info("HP Sim: Loading universe data into RAM...")
    all_data_df = disk_conn.execute("SELECT * FROM market_data").df()
    disk_conn.close()
    
    mem_conn = duckdb.connect(":memory:")
    mem_conn.register("market_data", all_data_df)
    
    class MemoryDataEngine:
        def __init__(self, conn): self.conn = conn
        def get_pit_view(self, ticker, as_of):
            query = f"SELECT * FROM market_data WHERE ticker = '{ticker}' AND event_time <= '{as_of}' ORDER BY event_time ASC"
            return self.conn.execute(query).df().set_index('event_time')

    data_engine = MemoryDataEngine(mem_conn)
    strategy = StrategyEngine(data_provider=data_engine, config_path="config.yaml")
    strategy.lab._conn = mem_conn
    
    # Range: Loaded from config.yaml
    tf = config['model_pipeline']['timeframes']
    
    def parse_date(d_str):
        if d_str == 'now': return datetime.now()
        return datetime.strptime(d_str, '%Y-%m-%d')
        
    start_date = parse_date(tf['train_start'])
    end_date = parse_date(tf['train_end'])
    
    logger.info(f"RL Pre-compute: Targeting window {start_date.date()} -> {end_date.date()}")
    
    current_time = start_date
    rankings_list = []
    prices_list = []
    
    # We rebalance weekly for training speed
    dates = []
    while current_time <= end_date:
        if current_time.weekday() == 0: # Monday only for RL simplicity in V1
            dates.append(current_time)
        current_time += timedelta(days=1)

    logger.info(f"Computing {len(dates)} rebalance snapshots...")
    
    for dt in tqdm(dates):
        view = strategy.get_current_rankings(as_of=dt)
        if view['status'] == "OK":
            # Rankings
            scores = {e['ticker']: e['score'] for e in view['ladder']}
            scores['date'] = dt
            rankings_list.append(scores)
            
            # Prices (we need these for the gym to calculate pnl)
            price_row = {'date': dt}
            for t in tickers:
                subset = all_data_df[(all_data_df['ticker'] == t) & (all_data_df['event_time'] <= dt)]
                if not subset.empty:
                    price_row[t] = float(subset.iloc[-1]['close'])
            prices_list.append(price_row)
            
        # GC
        if dt.day == 1:
            torch.cuda.empty_cache()
            strategy.lab._feat_cache = {}

    # Save to data/
    os.makedirs("data/rl", exist_ok=True)
    pd.DataFrame(rankings_list).set_index('date').to_csv("data/rl/train_rankings.csv")
    pd.DataFrame(prices_list).set_index('date').to_csv("data/rl/train_prices.csv")
    
    # Also save SPY separately
    spy_df = all_data_df[all_data_df['ticker'] == 'SPY'].set_index('event_time')
    spy_df.to_csv("data/rl/train_spy.csv")
    
    logger.success("✅ RL Data Pre-computation Complete!")

if __name__ == "__main__":
    precompute_rl_data()
