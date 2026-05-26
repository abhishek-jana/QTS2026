from datetime import datetime
import pandas as pd
import yaml
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.data_engine import DataEngine

with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
engine = DataEngine(storage_path='data/uqts_v2_intraday.ddb', read_only=True)
strategy = StrategyEngine(data_provider=engine, config_path='config.yaml')

db_last_str = engine.conn.execute("SELECT MAX(event_time) FROM market_data").fetchone()[0]
db_last_dt = pd.to_datetime(db_last_str)

tickers = config['universe']['tickers']
total_window = 63 + 250
days_back = max(total_window + 10, 100)
fetch_start = db_last_dt - pd.Timedelta(days=days_back)

bv1 = strategy.lab.get_batch_pit_view(tickers, db_last_dt, start_time=fetch_start)
bv2 = strategy.lab.get_batch_pit_view(tickers, db_last_dt, start_time=fetch_start)

diff = bv1.compare(bv2)
print("Differences:")
print(diff.head(10))

