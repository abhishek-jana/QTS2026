from datetime import datetime
import pandas as pd
import yaml
import torch
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.data_engine import DataEngine

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
engine = DataEngine(storage_path='data/uqts_v2_intraday.ddb', read_only=True)
strategy = StrategyEngine(data_provider=engine, config_path='config.yaml')

db_last_str = engine.conn.execute("SELECT MAX(event_time) FROM market_data").fetchone()[0]
db_last_dt = pd.to_datetime(db_last_str)

hv1 = strategy.get_current_rankings(db_last_dt)
hv2 = strategy.get_current_rankings(db_last_dt)

scores1 = {e['ticker']: e['score'] for e in hv1['ladder']}
scores2 = {e['ticker']: e['score'] for e in hv2['ladder']}

for k in scores1:
    if scores1[k] != scores2[k]:
        print(f"Mismatch {k}: {scores1[k]} vs {scores2[k]}")
else:
    print("Scores match perfectly across calls.")

