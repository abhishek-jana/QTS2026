import yaml
import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Protocol

from research_lab.alpha_universe import AlphaUniverse
from research_lab.alpha_labeler import AlphaLabeler
from research_lab.plugins.core_plugins import ModalityRegistry
from research_lab.real_data_ingestor import InstitutionalIngestor
from research_lab.alpha_ranker import MultiModalRankNet, InputSpec
from alpha_factory.meta_controller import BayesianMetaController
from qts_core.logger import logger

class IDataProvider(Protocol):
    def get_pit_view(self, ticker: str, as_of: datetime) -> pd.DataFrame: ...
    def get_batch_pit_view(self, tickers: List[str], as_of: datetime) -> pd.DataFrame: ...

class StrategyEngine:
    """
    Unified 'House View' Generator.
    Encapsulates Ingestion -> AlphaUniverse -> RankNet -> MetaController.
    """
    def __init__(self, data_provider: IDataProvider, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_provider = data_provider
        
        # 1. Initialize AlphaUniverse (Plugins are now auto-discovered by the Registry)
        self.lab = AlphaUniverse(data_provider=self.data_provider, config=self.config)

        # 2. Setup Ingestor
        self.ingestor = InstitutionalIngestor(self.data_provider)
        
        # 3. Initialize MetaController for belief scoring
        self.meta_controller = BayesianMetaController(
            prior_belief=self.config.get('risk_metacontroller', {}).get('prior_belief', 0.62)
        )
        
        # 4. Load the trained "Brain" from configurable path
        model_path = self.config.get('model_pipeline', {}).get('model_path', 'models/challenger_v1.pt')
        try:
            self.model = torch.jit.load(model_path)
            self.model.eval()
            logger.info(f"StrategyEngine: Loaded TorchScript RankNet ({model_path}).")
        except Exception as e:
            logger.warning(f"StrategyEngine: TorchScript load failed ({e}). Falling back to Python model.")
            # Standard triple-modality spec
            lookback = self.config['signal_physics'].get('lookback_days', 63)
            n_scales = 8
            specs = [
                InputSpec(name='x_seq', shape=(lookback, 1), type='seq'),
                InputSpec(name='x_spatial', shape=(1, n_scales, lookback), type='spatial'),
                InputSpec(name='x_graph', shape=(8,), type='graph')
            ]
            self.model = MultiModalRankNet(specs=specs, hidden_dim=64)
            self.model.eval()

    def ingest_data(self, tickers: List[str], start_date: str, end_date: str):
        """Dedicated method for manual ingestion."""
        if not self.ingestor:
             logger.warning("StrategyEngine: Ingestor not configured. Checking for existing data...")
             try:
                 test_view = self.data_provider.get_pit_view(tickers[0], datetime.now())
                 if test_view.empty:
                     logger.warning("StrategyEngine: No data found. Generating synthetic data for Mission Control UI...")
                     if hasattr(self.data_provider, 'generate_synthetic_pit_data'):
                         self.data_provider.generate_synthetic_pit_data(tickers)
                         logger.info("StrategyEngine: Synthetic data generation complete.")
                     else:
                         logger.error("StrategyEngine: Data provider does not support synthetic data generation.")
                 else:
                     logger.info(f"StrategyEngine: Found existing data ({len(test_view)} records for {tickers[0]}).")
             except Exception as e:
                 logger.error(f"StrategyEngine: Error checking for existing data: {e}")
             return
             
        logger.info(f"StrategyEngine: Ingesting {len(tickers)} tickers from {start_date} to {end_date}...")
        self.ingestor.ingest_universe(tickers, start_date, end_date)

    def get_current_rankings(self, as_of: datetime, include_batch: bool = False) -> Dict[str, Any]:
        """
        Returns a clean 'House View' of sorted tickers, scores, and signal energy.
        Aligns inference and metacognition using ALREADY INGESTED data.
        """
        tickers = self.config['universe']['tickers']
        
        # 2. Extract Point-in-Time Snapshot
        lookback = self.config.get('signal_physics', {}).get('lookback_days', 63)
        batch = self.lab.snapshot(as_of=as_of, tickers=tickers, lookback=lookback)
        
        if not batch:
            return {
                "ladder": [],
                "belief_score": self.meta_controller.get_position_scaler(),
                "signal_energy": 0.0,
                "as_of": as_of.isoformat(),
                "status": "DATA_MISSING"
            }
            
        # 3. Model Inference (RankNet LTR)
        with torch.no_grad():
            if hasattr(self.model, 'predict_dataset'):
                scores = self.model.predict_dataset(batch).squeeze()
            else:
                # Direct TorchScript Forward Pass
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in batch.data.items()}
                scores = self.model(inputs).squeeze()
            
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            # Ensure scores is at least 1D
            if np.isscalar(scores):
                scores = np.array([scores])
            elif scores.ndim == 0:
                scores = np.array([scores.item()])

        # 4. Calculate Signal Energy (Mean Absolute Power in the Spectrograms)
        signal_energy = float(torch.mean(torch.abs(batch.data['x_spatial'])).item())
        
        # 5. Build House View Ladder
        ladder = []
        for i, ticker in enumerate(batch.tickers):
            score_val = float(scores[i]) if len(scores) > i else float(scores[0])
            
            ladder.append({
                "ticker": ticker,
                "score": score_val,
                "price": float(batch.data.get('raw_price', torch.zeros(len(batch.tickers)))[i].item()),
                "energy": float(torch.mean(torch.abs(batch.data['x_spatial'][i])).item())
            })
            
        ladder.sort(key=lambda x: x['score'], reverse=True)
        
        view = {
            "ladder": ladder,
            "belief_score": self.meta_controller.get_position_scaler(),
            "signal_energy": signal_energy,
            "as_of": as_of.isoformat(),
            "status": "OK"
        }

        if include_batch:
            view["_batch"] = batch # Private reference for high-fidelity UI needs
            
        return view

    def update_model_metacognition(self, realized_returns: Dict[str, float], predicted_scores: Dict[str, float]):
        """
        Updates the Bayesian MetaController with realized performance.
        Allows the engine to 'learn' its own current validity.
        """
        tickers = list(realized_returns.keys())
        y_true = np.array([realized_returns[t] for t in tickers])
        y_pred = np.array([predicted_scores[t] for t in tickers])
        
        new_belief = self.meta_controller.update_belief(y_true, y_pred)
        logger.info(f"MetaController Updated: New Belief Score = {new_belief:.4f}")
        return new_belief
