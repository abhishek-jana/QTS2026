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
    all_data_df = all_data_df.reset_index()
    engine.close()
    
    # Pre-calculate rolling 21-day volatility for all tickers to save time
    logger.info("Calculating Universe Volatilities (Level 3 Risk Parity)...")
    vol_cache = {}
    for t in tickers:
        df_t = all_data_df[all_data_df['ticker'] == t].set_index('event_time').sort_index()
        vol_cache[t] = df_t['close'].pct_change().rolling(21).std()

    # Create strategy engine to load the model and universe
    strategy = StrategyEngine(data_provider=DataEngine(storage_path=db_path, read_only=True), config_path="config.yaml")
    
    tf = config['model_pipeline']['timeframes']
    
    def parse_date(d_str):
        if d_str == 'now': return datetime.now()
        return datetime.strptime(d_str, '%Y-%m-%d')
        
    start_date = parse_date(tf['train_start'])
    end_date = parse_date(tf['train_end'])
    
    logger.info(f"RL Pre-compute: Targeting window {start_date.date()} -> {end_date.date()}")
    
    # OPTIMIZATION: Use walk_forward to get pre-computed batches instantly
    steps = strategy.lab.walk_forward(universe=tickers, start_date=start_date, end_date=end_date, stride=1)
    
    if not steps:
        logger.error("❌ No steps returned from walk_forward.")
        return

    rankings_list = []
    prices_list = []
    vols_list = []
    
    device = next(strategy.model.parameters()).device
    
    logger.info(f"Computing {len(steps)} rebalance snapshots via cached batches...")
    
    for step in tqdm(steps):
        dt = step['date']
        batch = step['batch'].to(device)
        
        with torch.no_grad():
            tft_inputs = {
                'static': {k.replace('x_static_', ''): v for k, v in batch.data.items() if k.startswith('x_static_')},
                'past': {k.replace('x_past_', ''): v for k, v in batch.data.items() if k.startswith('x_past_')}
            }
            # Predict
            out_dict = strategy.model.model(tft_inputs)
            # Use median quantile (index 1) for ranking
            scores = out_dict['out'][:, 1].cpu().numpy()
            
        scores_map = {t: float(scores[i]) if i < len(scores) else 0.0 for i, t in enumerate(batch.tickers)}
        scores_map['date'] = dt
        rankings_list.append(scores_map)
        
        price_row = {'date': dt}
        vol_row = {'date': dt}
        
        for i, t in enumerate(batch.tickers):
            if 'raw_price' in batch.data:
                price_row[t] = float(batch.data['raw_price'][i].item())
            else:
                subset = all_data_df[(all_data_df['ticker'] == t) & (all_data_df['event_time'] <= dt)]
                price_row[t] = float(subset.iloc[-1]['close']) if not subset.empty else 0.0
            
            vol_series = vol_cache[t][vol_cache[t].index <= dt]
            vol_row[t] = float(vol_series.iloc[-1]) if not vol_series.empty and not pd.isna(vol_series.iloc[-1]) else 0.02
            
        prices_list.append(price_row)
        vols_list.append(vol_row)

    os.makedirs("data/rl", exist_ok=True)
    if not rankings_list:
        logger.error("❌ No valid rankings were generated.")
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