import asyncio
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cockpit_backend.streamer import DataStreamer

class MockManager:
    def __init__(self):
        self.active_connections = {}
    def subscribe(self, ws, ticker):
        self.active_connections[ws] = ticker
    def broadcast(self, msg):
        pass
    def send_to_subscribers(self, ticker, msg):
        pass

async def debug():
    manager = MockManager()
    streamer = DataStreamer(manager)
    
    # Mock some data
    class MockBatch:
        def __init__(self, tickers):
            self.tickers = tickers
            self.data = {
                'x_spatial': torch.randn(len(tickers), 1, 8, 63)
            }
    
    import torch
    batch = MockBatch(streamer.tickers)
    ticker = streamer.tickers[0]
    
    # We need a real DataEngine/StrategyEngine to test _get_spectral_data
    # Or at least mock the parts it uses.
    
    print(f"Testing spectral data for {ticker}")
    
    # Let's try to run the real initialization if possible, or just mock the data_provider
    try:
        # streamer.strategy.lab.data_provider.get_pit_view(ticker, self.current_knowledge_time)
        # Instead of full init, let's just mock the return of get_pit_view
        
        mock_history = pd.DataFrame({
            'open': np.random.rand(200),
            'high': np.random.rand(200),
            'low': np.random.rand(200),
            'close': np.random.rand(200)
        }, index=pd.date_range(end=datetime.now(), periods=200, freq='1h'))
        
        # Patch the strategy engine
        class MockLab:
            class MockDataProvider:
                def get_pit_view(self, t, kt):
                    return mock_history
            def __init__(self):
                self.data_provider = MockDataProvider()
        
        streamer.strategy.lab = MockLab()
        
        spectral_data = streamer._get_spectral_data(batch, ticker)
        
        history = spectral_data['history']
        print(f"History length: {len(history)}")
        if len(history) > 1:
            times = [h['time'] for h in history]
            is_sorted = all(times[i] <= times[i+1] for i in range(len(times)-1))
            has_duplicates = len(set(times)) < len(times)
            print(f"Sorted: {is_sorted}")
            print(f"Has duplicates: {has_duplicates}")
            if has_duplicates:
                from collections import Counter
                dupes = [item for item, count in Counter(times).items() if count > 1]
                print(f"Duplicate times: {dupes[:5]}")
                
    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug())
