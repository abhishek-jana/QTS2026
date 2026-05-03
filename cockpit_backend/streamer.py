import asyncio
import json
import numpy as np
import torch
from datetime import datetime, timedelta
from research_lab.alpha_universe import AlphaUniverse
from research_lab.plugins.core_plugins import SequentialPlugin, SpatialPlugin

class DataStreamer:
    def __init__(self, manager):
        self.manager = manager
        self.tickers = ["AAPL", "MSFT", "GOOG", "SPY"]
        self.lab = AlphaUniverse(plugins=[SequentialPlugin(), SpatialPlugin()])
        # Generate enough history
        self.lab.engine.generate_synthetic_pit_data(self.tickers, days=600)
        # Start at a date where we have enough lookback (63) and forward horizon (21)
        self.current_knowledge_time = datetime(2020, 5, 1) 

    async def start_streaming(self):
        """Simulates live market ticks and broadcasts to UI."""
        while True:
            if not self.manager.active_connections:
                await asyncio.sleep(2)
                continue

            # 1. Fetch current PIT snapshot
            batch = self.lab.snapshot(
                as_of=self.current_knowledge_time, 
                tickers=self.tickers, 
                lookback=63
            )

            if batch:
                # 2. Extract data for 5 panels
                payload = {
                    "timestamp": self.current_knowledge_time.isoformat(),
                    "spectral": self._get_spectral_data(batch),
                    "metacognition": self._get_metacognition_data(batch),
                    "rankings": self._get_ranking_data(batch),
                    "execution": self._get_execution_data(batch),
                    "pipeline": self._get_pipeline_data()
                }

                # 3. Broadcast
                await self.manager.broadcast(json.dumps(payload))

            # Move "time" forward by 1 hour per tick for the demo
            self.current_knowledge_time += timedelta(hours=1)
            await asyncio.sleep(0.5) # Fast scrolling

    def _get_spectral_data(self, batch):
        try:
            # Top ticker CWT
            ticker = self.tickers[0]
            ticker_indices = [i for i, t in enumerate(batch.tickers) if t == ticker]
            if not ticker_indices:
                return self._get_empty_spectral_data()
            
            idx = ticker_indices[-1]
            cwt_matrix = batch.data['x_spatial'][idx].squeeze().numpy()
            
            return {
                "ticker": ticker,
                "cwt": cwt_matrix.tolist(),
                "adf_p_value": 0.0001,
                "shap_values": {
                    "Momentum": np.random.uniform(0, 1),
                    "Sentiment": np.random.uniform(0, 1),
                    "Volatility": np.random.uniform(0, 1)
                }
            }
        except Exception as e:
            print(f"Error in spectral data generation: {e}")
            return self._get_empty_spectral_data()

    def _get_empty_spectral_data(self):
        return {
            "ticker": "WAITING",
            "cwt": np.zeros((8, 63)).tolist(),
            "adf_p_value": 1.0,
            "shap_values": {"N/A": 0}
        }

    def _get_metacognition_data(self, batch):
        return {
            "belief_score": 0.85 + np.random.normal(0, 0.02),
            "manifold_drift": np.random.randn(10, 2).tolist(), # Mock t-SNE points
            "alpha_decay": (np.cumsum(np.random.uniform(0, 0.1, 30))).tolist()
        }

    def _get_ranking_data(self, batch):
        # Decile ladder logic
        sorted_indices = np.argsort(batch.labels.numpy())[::-1]
        ladder = []
        for i in sorted_indices:
            ladder.append({
                "ticker": batch.tickers[i],
                "score": float(batch.labels[i])
            })
        
        return {
            "ladder": ladder,
            "ls_spread": np.sin(np.linspace(0, 10, 100)).tolist() # Mock equity curve
        }

    def _get_execution_data(self, batch):
        return {
            "implementation_shortfall": np.random.uniform(0, 5), # Bps
            "slippage_heatmap": np.random.rand(5, 5).tolist()
        }

    def _get_pipeline_data(self):
        return {
            "champion_sharpe": 1.8,
            "challenger_sharpe": 2.1,
            "training_progress": "Epoch 42: Loss 0.0031... Validation IC: 0.05"
        }
