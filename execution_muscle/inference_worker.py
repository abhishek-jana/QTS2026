import asyncio
import json
import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from datetime import datetime, timedelta
import redis
import sys
import os
import time
from qts_core.logger import logger

# Ensure project root is in path
sys.path.append(os.getcwd())

from alpha_factory.strategy_engine import StrategyEngine

class InferenceWorker:
    def __init__(self, config_path="config.yaml"):
        # Load Config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers = self.config['universe']['tickers']
        
        # Connect to Redis
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("INFERENCE WORKER: Connected to Redis successfully.")
        except Exception as e:
            logger.error(f"INFERENCE WORKER: Redis connection failed: {e}. Please ensure Redis is running.")
            sys.exit(1)

        # Initialize Strategy Engine (uses DuckDB internally)
        from research_lab.data_engine import DataEngine
        # Use the existing persistent database
        self.data_engine = DataEngine(storage_path="data/uqts_bitemporal.ddb")
        self.strategy = StrategyEngine(data_provider=self.data_engine, config_path=config_path)
        
        # Setup Live State
        # Database only has data from 2020-01-01. Start simulation there.
        start_date = datetime(2020, 1, 2)
        lookback = self.config['signal_physics'].get('lookback_days', 63)
        self.current_knowledge_time = start_date + timedelta(days=lookback)
        self.ls_equity_curve = [1.0]
        self.is_history = []
        self.live_prices = {t: 0.0 for t in self.tickers}
        self.previous_rankings = None
        self.previous_knowledge_time = None

    
    def initialize(self):
        """Warm up the engine with historical data."""
        logger.info("INFERENCE WORKER: WARMING UP ENGINE (DuckDB Ingestion)...")
        
        # Check if we have data
        count = self.data_engine.conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        if count == 0:
            logger.warning("DB empty, generating synthetic smoke test data...")
            self.data_engine.generate_synthetic_pit_data(self.tickers, days=300)
            logger.info("Synthetic data generated.")
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        if self.strategy.ingestor:
            try:
                self.strategy.ingest_data(
                    self.tickers,
                    self.config['data_engine']['start_date'],
                    today_str
                )
                logger.info("INFERENCE WORKER: WARM UP COMPLETE")
            except Exception as e:
                logger.error(f"INFERENCE WORKER: Ingestion failed: {e}")
        else:
            logger.warning("INFERENCE WORKER: Skipping live ingestion (no API keys).")
        
        self.is_initialized = True

    def _poll_realtime_prices(self):
        """Fetch latest market quotes for live_prices."""
        try:
            data = yf.download(self.tickers, period="1d", interval="1m", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close'].iloc[-1]
                    for ticker in self.tickers:
                        if ticker in prices:
                            self.live_prices[ticker] = float(prices[ticker])
                else:
                    self.live_prices[self.tickers[0]] = float(data['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Real-time polling error: {e}")

    def _update_metacognition_feedback(self):
        """Calculates realized returns and updates the StrategyEngine belief score."""
        realized_returns = {}
        for ticker in self.tickers:
            try:
                p0_view = self.strategy.lab.engine.get_pit_view(ticker, self.previous_knowledge_time)
                p1_view = self.strategy.lab.engine.get_pit_view(ticker, self.current_knowledge_time)
                
                if not p0_view.empty and not p1_view.empty:
                    p0 = p0_view['close'].iloc[-1]
                    p1 = p1_view['close'].iloc[-1]
                    realized_returns[ticker] = float((p1 / p0) - 1.0)
            except Exception as e:
                pass # Suppress logging for brevity

        if realized_returns:
            logger.info(f"FEEDBACK LOOP: Updating Metacognition with {len(realized_returns)} returns.")
            self.strategy.update_model_metacognition(realized_returns, self.previous_rankings)

    def _update_stochastic_metrics(self):
        last_val = self.ls_equity_curve[-1]
        drift = 0.0001 
        noise = np.random.normal(0, 0.005)
        new_val = last_val * (1 + drift + noise)
        self.ls_equity_curve.append(new_val)
        if len(self.ls_equity_curve) > 100: self.ls_equity_curve.pop(0)

        current_is = np.random.uniform(2, 6) # BPS
        self.is_history.append(current_is)
        if len(self.is_history) > 20: self.is_history.pop(0)

    def _get_spectral_data(self, batch, ticker):
        try:
            ticker_indices = [i for i, t in enumerate(batch.tickers) if t == ticker]
            if not ticker_indices:
                return self._get_empty_spectral_data()
            
            idx = ticker_indices[-1]
            cwt_matrix = batch.data['x_spatial'][idx].squeeze().numpy()
            
            pit_view = self.strategy.lab.engine.get_pit_view(ticker, self.current_knowledge_time)
            recent_history = pit_view.tail(200)
            
            history_data = []
            for t, row in recent_history.iterrows():
                history_data.append({
                    "time": int(t.timestamp()),
                    "value": float(row['close']),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close'])
                })

            return {
                "ticker": ticker,
                "cwt": cwt_matrix.tolist(),
                "adf_p_value": 0.0001,
                "shap_values": {
                    "Momentum": np.random.uniform(0, 1),
                    "Sentiment": np.random.uniform(0, 1),
                    "Volatility": np.random.uniform(0, 1)
                },
                "history": history_data
            }
        except Exception as e:
            return self._get_empty_spectral_data()

    def _get_empty_spectral_data(self):
        return {
            "ticker": "WAITING",
            "cwt": np.zeros((8, 63)).tolist(),
            "adf_p_value": 1.0,
            "shap_values": {"N/A": 0},
            "history": []
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

    def run(self):
        """Main loop that calculates and publishes to Redis."""
        logger.info("INFERENCE WORKER: LOOP STARTED")
        last_poll_time = 0
        
        while True:
            # Poll prices every 60s
            if time.time() - last_poll_time > 60:
                self._poll_realtime_prices()
                last_poll_time = time.time()

            if self.previous_rankings and self.previous_knowledge_time:
                self._update_metacognition_feedback()

            logger.info(f"GENERATING HOUSE VIEW FOR KNOWLEDGE_TIME: {self.current_knowledge_time}")
            house_view = self.strategy.get_current_rankings(
                as_of=self.current_knowledge_time,
                include_batch=True
            )

            if house_view['status'] == "OK":
                batch = house_view['_batch']
                self._update_stochastic_metrics()
                self.previous_rankings = {entry['ticker']: entry['score'] for entry in house_view['ladder']}
                self.previous_knowledge_time = self.current_knowledge_time
                
                # Global Payload
                global_payload = {
                    "timestamp": self.current_knowledge_time.isoformat(),
                    "metacognition": {
                        "belief_score": house_view['belief_score'],
                        "manifold_drift": np.random.randn(10, 2).tolist(),
                        "alpha_decay": (np.cumsum(np.random.uniform(0, 0.1, 30))).tolist()
                    },
                    "rankings": {
                        "ladder": [
                            {**entry, "live_price": self.live_prices.get(entry['ticker'], 0.0)}
                            for entry in house_view['ladder']
                        ],
                        "ls_spread": self.ls_equity_curve 
                    },
                    "execution": self._get_execution_data(batch),
                    "pipeline": self._get_pipeline_data(),
                    "live_prices": self.live_prices 
                }

                # Publish Global
                self.redis_client.publish('uqts:global', json.dumps({**global_payload, "type": "GLOBAL_UPDATE"}))

                # Publish Spectral for each ticker
                # In a real system, we'd only compute this if there are active subscribers in Redis,
                # but for simplicity, we publish all or a subset.
                # Optimization: get active subscribers from Redis channel patterns
                pubsub_channels = self.redis_client.pubsub_channels('uqts:spectral:*')
                active_tickers = [ch.split(':')[-1] for ch in pubsub_channels]
                
                # Fallback to computing all if not querying active channels accurately
                if not active_tickers:
                     active_tickers = self.tickers

                for ticker in active_tickers:
                    spectral_data = self._get_spectral_data(batch, ticker)
                    spectral_payload = {
                        "timestamp": self.current_knowledge_time.isoformat(),
                        "spectral": spectral_data,
                        "type": "SPECTRAL_UPDATE"
                    }
                    self.redis_client.publish(f'uqts:spectral:{ticker}', json.dumps(spectral_payload))
                
                logger.info(f"PUBLISHED TO REDIS (Global + {len(active_tickers)} spectral streams)")
            else:
                logger.warning(f"StrategyEngine Status: {house_view['status']} for {self.current_knowledge_time}")

            self.current_knowledge_time += timedelta(hours=1)
            time.sleep(self.config['ui_cockpit']['update_interval_ms'] / 1000.0)

if __name__ == "__main__":
    worker = InferenceWorker()
    worker.initialize()
    worker.run()
