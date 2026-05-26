from datetime import datetime
import pandas as pd
import yaml
import torch
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.data_engine import DataEngine

with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
engine = DataEngine(storage_path='data/uqts_v2_intraday.ddb', read_only=True)
strategy = StrategyEngine(data_provider=engine, config_path='config.yaml')

db_last_str = engine.conn.execute("SELECT MAX(event_time) FROM market_data").fetchone()[0]
db_last_dt = pd.to_datetime(db_last_str)

tickers = config['universe']['tickers']

batch1 = strategy.lab.snapshot(as_of=db_last_dt, tickers=tickers, require_labels=False)
batch2 = strategy.lab.snapshot(as_of=db_last_dt, tickers=tickers, require_labels=False)

diffs = 0
for k in batch1.data.keys():
    v1 = batch1.data[k]
    v2 = batch2.data[k]
    # Check ignoring NaNs
    mask = ~torch.isnan(v1) & ~torch.isnan(v2)
    if not torch.allclose(v1[mask], v2[mask]):
        print(f"Input mismatch in {k}!")
        diffs += 1

print(f"Total diffs: {diffs}")
