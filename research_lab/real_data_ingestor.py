import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from research_lab.data_engine import DataEngine

class YFinanceIngestor:
    """
    Ingests real US Equities data from Yahoo Finance into the PIT DataEngine.
    """
    def __init__(self, data_engine: DataEngine):
        self.engine = data_engine

    def ingest_universe(self, tickers: list, start_date: str, end_date: str):
        """
        Downloads data and injects it into the engine with bi-temporal safety.
        Assumes Daily Knowledge Time = Event Time + 16:00 (market close).
        """
        print(f"📥 DOWNLOADING REAL MARKET DATA: {tickers} from {start_date} to {end_date}...")
        
        all_data = []
        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                continue
            
            # Flatten MultiIndex columns: ('Open', 'AAPL') -> 'Open'
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df = df.reset_index()
            for _, row in df.iterrows():
                event_time = row['Date']
                # Knowledge Time: For research, we assume we know the day's close at 16:00
                knowledge_time = event_time + timedelta(hours=16)
                
                all_data.append({
                    'ticker': ticker,
                    'event_time': event_time,
                    'knowledge_time': knowledge_time,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
        
        # Inject into DataEngine
        self.engine.registry = pd.DataFrame(all_data)
        print(f"✅ INGESTION COMPLETE: {len(self.engine.registry)} real-market records loaded.")

if __name__ == "__main__":
    engine = DataEngine()
    ingestor = YFinanceIngestor(engine)
    # Test with expanded high-beta tech universe
    ingestor.ingest_universe(['AAPL', 'MSFT', 'GOOG', 'SPY', 'AMZN', 'NFLX', 'META', 'NVDA'], '2022-01-01', '2024-01-01')

    
    # Verify PIT view
    view = engine.get_pit_view('AAPL', datetime(2023, 1, 1))
    print(f"Sample PIT Data (AAPL): {len(view)} days known as of 2023-01-01.")
