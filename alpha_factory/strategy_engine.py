import yaml
import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Protocol

from research_lab.alpha_universe import AlphaUniverse, MultiModalBatch
from research_lab.alpha_labeler import AlphaLabeler
from research_lab.plugins.core_plugins import ModalityRegistry
from research_lab.real_data_ingestor import InstitutionalIngestor
from research_lab.alpha_ranker_sniper import SniperRanker
from alpha_factory.meta_controller import BayesianMetaController
from qts_core.logger import logger

class IDataProvider(Protocol):
    def get_pit_view(self, ticker: str, as_of: datetime) -> pd.DataFrame: ...
    def get_batch_pit_view(self, tickers: List[str], as_of: datetime) -> pd.DataFrame: ...

class StrategyEngine:
    """
    Unified 'House View' Generator (Sniper V7.0).
    Encapsulates Ingestion -> AlphaUniverse -> SniperRanker (TFT) -> MetaController.
    """
    def __init__(self, data_provider: IDataProvider, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_provider = data_provider
        
        # 1. Initialize AlphaUniverse (Plugins are now auto-discovered by the Registry)
        self.lab = AlphaUniverse(conn=self.data_provider.conn, config=self.config)
        self.lab.labeler = AlphaLabeler(mode='directional')

        # 2. Setup Ingestor
        self.ingestor = InstitutionalIngestor(self.data_provider, config=self.config)
        
        # 3. Initialize MetaController for belief scoring
        self.meta_controller = BayesianMetaController(
            prior_belief=self.config.get('risk_metacontroller', {}).get('prior_belief', 0.75)
        )
        
        # 4. Define model specs for TFT
        lookback = self.config['signal_physics'].get('lookback_days', 63)
        n_scales = len(self.config['signal_physics']['wavelet_transform']['scales'])
        
        self.specs = {
            'static': {'x_static': 1},
            'past': {
                'x_seq': 1,
                'x_spatial': n_scales,
                'x_volume': 1,
                'x_momentum': 3,
                'x_calendar': 4
            }
        }
        self.past_keys = sorted(self.specs['past'].keys())

        # 5. Load the trained "Sniper"
        model_path = self.config.get('model_pipeline', {}).get('model_path', 'models/sniper_v7.pt')
        hidden_dim = self.config.get('model_pipeline', {}).get('architecture', {}).get('hidden_dim', 32)
        
        self.model = SniperRanker(specs=self.specs, hidden_dim=hidden_dim)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                logger.info(f"StrategyEngine: Loaded SniperRanker ({model_path}).")
            except Exception as e:
                logger.warning(f"StrategyEngine: Model load failed ({e}). Using uninitialized model.")
        else:
            logger.warning(f"StrategyEngine: Model file {model_path} not found. Using uninitialized model.")

    def ingest_data(self, tickers: List[str], start_date: str, end_date: str):
        """Dedicated method for manual ingestion."""
        if not self.ingestor:
             logger.warning("StrategyEngine: Ingestor not configured.")
             return
        self.ingestor.ingest_universe(tickers, start_date, end_date)

    def get_current_rankings(self, as_of: datetime, include_batch: bool = False) -> Dict[str, Any]:
        """
        Returns a clean 'House View' of sorted tickers, scores, and signal energy.
        Aligns inference and metacognition using ALREADY INGESTED data.
        """
        tickers = self.config['universe']['tickers']
        
        # Extract Point-in-Time Snapshot
        lookback = self.config.get('signal_physics', {}).get('lookback_days', 63)
        # Ferrari Fix: require_labels=False for live inference
        batch = self.lab.snapshot(as_of=as_of, tickers=tickers, lookback=lookback, require_labels=False)
        
        if not batch:
            return {
                "ladder": [],
                "belief_score": self.meta_controller.get_position_scaler(),
                "signal_energy": 0.0,
                "as_of": as_of.isoformat(),
                "status": "DATA_MISSING"
            }
            
        # Model Inference (Sniper TFT)
        with torch.no_grad():
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in batch.data.items()}

            tft_inputs = {
                'static': {k.replace('x_static_', ''): v for k, v in inputs.items() if k.startswith('x_static_')},
                'past': {k.replace('x_past_', ''): v for k, v in inputs.items() if k.startswith('x_past_')}
            }
            
            # Predict
            out_dict = self.model.model(tft_inputs)
            
            # Use median quantile (index 1) for ranking
            scores = out_dict['out'][:, 1].cpu().numpy()
            
            # Average VSN weights over the sequence length for interpretability
            past_weights = out_dict['past_weights'].mean(dim=1).cpu().numpy() # [batch, n_vars]

            # Vectorized Signal Energy (Spatial Wavelet mean magnitude)
            energy_all = torch.zeros(len(batch.tickers))
            if 'x_past_x_spatial' in batch.data:
                # Ferrari robust reduction: mean everything after the batch dimension
                energy_all = torch.mean(torch.abs(batch.data['x_past_x_spatial']).flatten(1), dim=1).cpu()

        # Build House View Ladder
        ladder = []
        for i, ticker in enumerate(batch.tickers):
            score_val = float(scores[i]) if len(scores) > i else 0.0
            
            # Map weights to human names
            shap = {}
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'past_keys'):
                for j, p_name in enumerate(self.model.model.past_keys):
                    shap[p_name] = float(past_weights[i, j])
            else:
                # Fallback if using a mock/uninitialized model
                for j, p_name in enumerate([k for k in self.past_keys if k != 'x_spatial']):
                    if j < past_weights.shape[1]:
                        shap[p_name] = float(past_weights[i, j])
            
            # Try to grab raw price if it was included
            raw_price = 0.0
            if 'raw_price' in batch.data:
                raw_price = float(batch.data['raw_price'][i].item())
                
            # Grab pre-calculated energy
            energy = float(energy_all[i].item())
            
            ladder.append({
                "ticker": ticker,
                "score": score_val,
                "price": raw_price,
                "energy": energy,
                "shap": shap
            })
            
        ladder.sort(key=lambda x: x['score'], reverse=True)
        
        # Spatial energy as proxy for signal energy
        global_energy = 0.0
        if 'x_past_x_spatial' in batch.data:
            global_energy = float(torch.mean(torch.abs(batch.data['x_past_x_spatial'])).item())
        
        view = {
            "ladder": ladder,
            "belief_score": self.meta_controller.get_position_scaler(),
            "signal_energy": global_energy,
            "as_of": as_of.isoformat(),
            "status": "OK"
        }

        if include_batch:
            view["_batch"] = batch
            
        return view

    def get_ticker_diagnostics(self, ticker: str, as_of: datetime) -> Optional[Dict[str, Any]]:
        """Provides deep spectral insights for a specific ticker for UI visualization."""
        lookback = self.config.get('signal_physics', {}).get('lookback_days', 63)
        # Ferrari Fix: require_labels=False for live diagnostics
        batch = self.lab.snapshot(as_of=as_of, tickers=[ticker], lookback=lookback, require_labels=False)
        if not batch: return None
        
        # 1. Extract History
        raw_prices = self.lab.get_batch_pit_view([ticker], as_of, start_time=as_of - timedelta(days=lookback))
        history_ui = []
        if not raw_prices.empty:
            for t, row in raw_prices.iterrows():
                history_ui.append({
                    "time": int(t.timestamp()),
                    "open": float(row['open']), "high": float(row['high']),
                    "low": float(row['low']), "close": float(row['close']), "volume": int(row['volume'])
                })

        # 2. Extract Wavelet CWT
        cwt = np.zeros((16, lookback))
        if 'x_past_x_spatial' in batch.data:
            cwt = batch.data['x_past_x_spatial'][0].cpu().numpy().T # Shape: (scales, seq)
        
        # 3. STATISTICAL DIAGNOSTICS (REAL ADF TEST)
        adf_p = 0.99
        try:
            from statsmodels.tsa.stattools import adfuller
            if 'x_past_x_seq' in batch.data:
                series = batch.data['x_past_x_seq'][0].squeeze().cpu().numpy()
                if len(series) > 10:
                    res = adfuller(series, autolag='AIC')
                    adf_p = float(res[1])
        except Exception as e:
            logger.warning(f"ADF Test failed for {ticker}: {e}")
            if 'x_past_x_seq' in batch.data:
                adf_p = float(torch.std(batch.data['x_past_x_seq']).item()) # Fallback proxy
        
        # 4. Model weights (SHAP proxy from VSN)
        view = self.get_current_rankings(as_of)
        shap = {}
        for entry in view.get('ladder', []):
            if entry['ticker'] == ticker:
                shap = entry['shap']
                break
        
        return {
            "ticker": ticker,
            "history": history_ui,
            "cwt": cwt,
            "adf_p": adf_p,
            "shap_fusion": shap
        }

    def update_model_metacognition(self, realized_returns: Dict[str, float], predicted_scores: Dict[str, float]):
        """Updates Bayesian MetaController."""
        tickers = list(realized_returns.keys())
        y_true = np.array([realized_returns[t] for t in tickers])
        y_pred = np.array([predicted_scores[t] for t in tickers])
        
        new_belief = self.meta_controller.update_belief(y_true, y_pred)
        return new_belief
