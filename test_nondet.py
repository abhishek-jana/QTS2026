import torch
import pandas as pd
from research_lab.data_engine import DataEngine
from alpha_factory.strategy_engine import StrategyEngine
import yaml

with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
engine = DataEngine(storage_path='data/uqts_v2_intraday.ddb', read_only=True)
strategy = StrategyEngine(data_provider=engine, config_path='config.yaml')
strategy.model.eval()

db_last_str = engine.conn.execute("SELECT MAX(event_time) FROM market_data").fetchone()[0]
db_last_dt = pd.to_datetime(db_last_str)
batch = strategy.lab.snapshot(as_of=db_last_dt, tickers=['TSLA', 'MRK', 'MSFT', 'HD'], require_labels=False)

with torch.no_grad():
    out1 = strategy.model(batch)
    out2 = strategy.model(batch)

print("Output 1:", out1[:, 1])
print("Output 2:", out2[:, 1])
print("Are they equal?", torch.allclose(out1, out2))
