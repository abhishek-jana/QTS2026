import yaml
import pandas as pd
from datetime import datetime
from research_lab.data_engine import DataEngine
from alpha_factory.strategy_engine import StrategyEngine

with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
engine = DataEngine(storage_path='data/uqts_v2_intraday.ddb', read_only=True)
strategy = StrategyEngine(data_provider=engine, config_path='config.yaml')

db_last_str = engine.conn.execute("SELECT MAX(event_time) FROM market_data").fetchone()[0]
db_last_dt = pd.to_datetime(db_last_str)

print("First call...")
v1 = strategy.get_current_rankings(db_last_dt)
print("Second call...")
v2 = strategy.get_current_rankings(db_last_dt)
print(f"Are they the same object? {v1 is v2}")
