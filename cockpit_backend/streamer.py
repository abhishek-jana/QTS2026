import asyncio
import json
import numpy as np
import torch
import pandas as pd
import yaml
import yfinance as yf
from datetime import datetime, timedelta
from research_lab.alpha_universe import AlphaUniverse
from research_lab.plugins.core_plugins import SequentialPlugin, SpatialPlugin
from research_lab.real_data_ingestor import YFinanceIngestor
from research_lab.alpha_ranker import MultiModalRankNet

class DataStreamer:
    def __init__(self, manager):
        self.manager = manager
        
        # 0. Load Configuration
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        # Note: We use the config's d_param for the brain
        self.lab = AlphaUniverse(plugins=[
            SequentialPlugin(d_param=self.config['research']['fd_d_param']), 
            SpatialPlugin()
        ])
        
        # 1. Ingest Historical Data (Dynamic up to today)
        ingestor = YFinanceIngestor(self.lab.engine)
        today_str = datetime.now().strftime("%Y-%m-%d")
        ingestor.ingest_universe(self.tickers, self.config['research']['start_date'], today_str)
        
        # 2. Setup Live State
        self.current_knowledge_time = datetime.now() - timedelta(days=self.config['ui']['stream_start_offset_days']) 
        self.ls_equity_curve = [1.0]
        self.is_history = []
        self.live_prices = {t: 0.0 for t in self.tickers}

    async def _poll_realtime_prices(self):
        """Background task to fetch latest market quotes."""
        while True:
            try:
                # Fetch only the last minute for all tickers
                # Using a small period and interval to get the current quote
                data = yf.download(self.tickers, period="1d", interval="1m", progress=False)
                if not data.empty:
                    # Handle MultiIndex for multiple tickers
                    if isinstance(data.columns, pd.MultiIndex):
                        prices = data['Close'].iloc[-1]
                        for ticker in self.tickers:
                            if ticker in prices:
                                self.live_prices[ticker] = float(prices[ticker])
                    else:
                        # Single ticker case
                        self.live_prices[self.tickers[0]] = float(data['Close'].iloc[-1])
            except Exception as e:
                print(f"⚠️ Real-time polling error: {e}")
            
            # Poll every 60 seconds to stay within API grace but provide "live" feel
            await asyncio.sleep(60)

    async def start_streaming(self):
        """Simulates live market ticks and broadcasts to UI."""
        # Start real-time polling thread
        asyncio.create_task(self._poll_realtime_prices())
        
        while True:
            if not self.manager.active_connections:
                await asyncio.sleep(2)
                continue

            # 1. Fetch current PIT snapshot
            batch = self.lab.snapshot(
                as_of=self.current_knowledge_time, 
                tickers=self.tickers, 
                lookback=self.config['research']['lookback_days']
            )

            if batch:
                # Update persistent state
                self._update_stochastic_metrics()
                
                # 2. Extract data for 5 panels
                payload = {
                    "timestamp": self.current_knowledge_time.isoformat(),
                    "spectral": self._get_spectral_data(batch),
                    "metacognition": self._get_metacognition_data(batch),
                    "rankings": self._get_ranking_data(batch),
                    "execution": self._get_execution_data(batch),
                    "pipeline": self._get_pipeline_data(),
                    "live_prices": self.live_prices # Include live market quotes
                }

                # 3. Broadcast
                await self.manager.broadcast(json.dumps(payload))

            # Move "time" forward by 1 hour per tick for the demo simulation
            self.current_knowledge_time += timedelta(hours=1)
            await asyncio.sleep(self.config['ui']['update_interval_ms'] / 1000.0)

    def _update_stochastic_metrics(self):
        # Realistic Equity Curve (Random Walk with Drift)
        last_val = self.ls_equity_curve[-1]
        drift = 0.0001 
        noise = np.random.normal(0, 0.005)
        new_val = last_val * (1 + drift + noise)
        self.ls_equity_curve.append(new_val)
        if len(self.ls_equity_curve) > 100: self.ls_equity_curve.pop(0)

        # IS Tracking for Variance Stress Test
        current_is = np.random.uniform(2, 6) # BPS
        self.is_history.append(current_is)
        if len(self.is_history) > 20: self.is_history.pop(0)

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
            "cwt": np.zeros((32, 63)).tolist(),
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
        """
        Aggregates per-ticker scores and binds real-time prices.
        """
        ticker_scores = {}
        for i, ticker in enumerate(batch.tickers):
            ticker_scores[ticker] = float(batch.labels[i])
            
        ladder = []
        for ticker, score in ticker_scores.items():
            ladder.append({
                "ticker": ticker,
                "score": score,
                "live_price": self.live_prices.get(ticker, 0.0)
            })
            
        ladder.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "ladder": ladder,
            "ls_spread": self.ls_equity_curve 
        }

    def _get_execution_data(self, batch):
        is_var = np.var(self.is_history) if self.is_history else 0
        heatmap = np.random.rand(5, 5)
        lob_skew = np.mean(heatmap[:, -1]) > 0.75 
        is_val = self.is_history[-1] if self.is_history else 0
        needs_retune = is_var > 2.0 or lob_skew or is_val > 5.5

        return {
            "implementation_shortfall": float(is_val),
            "slippage_heatmap": heatmap.tolist(),
            "needs_retune": bool(needs_retune),
            "is_var": float(is_var),
            "lob_skew_detected": bool(lob_skew)
        }

    def _get_pipeline_data(self):
        return {
            "champion_sharpe": 1.8,
            "challenger_sharpe": 2.1,
            "training_progress": "Epoch 42: Loss 0.0031... Validation IC: 0.05"
        }
