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
    logger.info("🚀 Pre-computing High-Fidelity RL Training Data (Sensor V5.0 + Level 3 Risk Parity)...")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    db_path = config['data_engine']['storage_path']
    tickers = config['universe']['tickers']
    
    # SENIOR FIX: Use DataEngine to handle connections and locks resiliently
    engine = DataEngine(storage_path=db_path, read_only=True)
    all_data_df = engine.get_batch_pit_view(tickers + ['SPY'], as_of=datetime.now())
    # SENIOR FIX: Reset index so 'event_time' is a column for registration/pivoting
    all_data_df = all_data_df.reset_index()
    engine.close()
    
    mem_conn = duckdb.connect(":memory:")
    # Register the snapshot in memory for fast RL sensor calculation
    mem_conn.register("market_data", all_data_df)
    
    class MemoryDataEngine:
        def __init__(self, conn): self.conn = conn
        def get_pit_view(self, ticker, as_of):
            query = f"SELECT * FROM market_data WHERE ticker = '{ticker}' AND event_time <= '{as_of}' ORDER BY event_time ASC"
            return self.conn.execute(query).df().set_index('event_time')

    data_engine = MemoryDataEngine(mem_conn)
    strategy = StrategyEngine(data_provider=data_engine, config_path="config.yaml")
    strategy.lab._conn = mem_conn
    
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
    vols_list = [] # LEVEL 3: Track volatilities
    
    # Pre-calculate rolling 21-day volatility for all tickers to save time
    logger.info("Calculating Universe Volatilities (Level 3 Risk Parity)...")
    vol_cache = {}
    for t in tickers:
        df_t = all_data_df[all_data_df['ticker'] == t].set_index('event_time').sort_index()
        vol_cache[t] = df_t['close'].pct_change().rolling(21).std()

    dates = []
    while current_time <= end_date:
        if current_time.weekday() == 0: 
            dates.append(current_time)
        current_time += timedelta(days=1)

    logger.info(f"Computing {len(dates)} rebalance snapshots...")
    
    for dt in tqdm(dates):
        view = strategy.get_current_rankings(as_of=dt)
        if view['status'] == "OK":
            scores = {e['ticker']: e['score'] for e in view['ladder']}
            scores['date'] = dt
            rankings_list.append(scores)
            
            price_row = {'date': dt}
            vol_row = {'date': dt}
            for t in tickers:
                subset = all_data_df[(all_data_df['ticker'] == t) & (all_data_df['event_time'] <= dt)]
                if not subset.empty:
                    price_row[t] = float(subset.iloc[-1]['close'])
                # Get closest vol measurement
                vol_series = vol_cache[t][vol_cache[t].index <= dt]
                vol_row[t] = float(vol_series.iloc[-1]) if not vol_series.empty and not pd.isna(vol_series.iloc[-1]) else 0.02 # Default 2% daily vol fallback
                
            prices_list.append(price_row)
            vols_list.append(vol_row)
            
        if dt.day == 1:
            torch.cuda.empty_cache()
            strategy.lab._feat_cache = {}

    os.makedirs("data/rl", exist_ok=True)
    if not rankings_list:
        logger.error("❌ No valid rankings were generated. Check for data gaps in AlphaUniverse.")
        return
        
    pd.DataFrame(rankings_list).set_index('date').to_csv("data/rl/train_rankings.csv")
    pd.DataFrame(prices_list).set_index('date').to_csv("data/rl/train_prices.csv")
    pd.DataFrame(vols_list).set_index('date').to_csv("data/rl/train_stock_vols.csv")
    
    # --- MACRO ENHANCEMENT: train_spy.csv ---
    spy_df = all_data_df[all_data_df['ticker'] == 'SPY'].set_index('event_time').sort_index()
    spy_df['ret'] = spy_df['close'].pct_change()
    
    # 1. Vol-of-Vol (VVIX Proxy)
    spy_df['vol_21'] = spy_df['ret'].rolling(21).std()
    spy_df['vov_21'] = spy_df['vol_21'].rolling(21).std()
    
    # 2. Institutional MA Crossovers
    spy_df['ma_50'] = spy_df['close'].rolling(50).mean()
    spy_df['ma_200'] = spy_df['close'].rolling(200).mean()
    spy_df['ma_ratio'] = spy_df['ma_50'] / spy_df['ma_200']
    
    # 3. Momentum RSI Proxy
    delta = spy_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    spy_df['rsi_14'] = 100 - (100 / (1 + rs))
    
    spy_df.ffill().fillna(0).to_csv("data/rl/train_spy.csv")
    
    logger.success("✅ RL Data Pre-computation Complete with Macro Sensors and Level 3 Volatilities!")

if __name__ == "__main__":
    precompute_rl_data()
