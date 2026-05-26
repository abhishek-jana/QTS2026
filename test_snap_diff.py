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

for i in range(len(batch1.tickers)):
    t1 = batch1.data['x_past_x_seq'][i]
    t2 = batch2.data['x_past_x_seq'][i]
    if not torch.allclose(t1, t2):
        print(f"Diff in {batch1.tickers[i]}:")
        print(t1[:5])
        print(t2[:5])
        break

