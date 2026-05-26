from research_lab.data_engine import DataEngine
from datetime import datetime
import pandas as pd

engine = DataEngine(storage_path='data/uqts_v2_intraday.ddb', read_only=True)
db_last_str = engine.conn.execute("SELECT MAX(event_time) FROM market_data").fetchone()[0]
now = datetime.now()
db_last_dt = pd.to_datetime(db_last_str) if db_last_str else now
diff = (now - db_last_dt).total_seconds()
print(f"Now: {now}, DB Last: {db_last_dt}, Diff: {diff}")
