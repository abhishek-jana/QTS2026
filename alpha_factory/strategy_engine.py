import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
import torch
import yaml

from alpha_factory.meta_controller import BayesianMetaController
from qts_core.logger import logger
from research_lab.alpha_labeler import AlphaLabeler
from research_lab.alpha_ranker_sniper import SniperRanker
from research_lab.alpha_universe import AlphaUniverse, MultiModalBatch
from research_lab.plugins.core_plugins import ModalityRegistry
from research_lab.real_data_ingestor import InstitutionalIngestor


class IDataProvider(Protocol):
    """
    Minimal contract for data providers consumed by the StrategyEngine.

    `conn` is part of the contract because AlphaUniverse needs a live
    DuckDB connection for PIT view materialization. Implementations that
    cannot expose a real connection should provide a duck-typed object
    that supports the SQL surface AlphaUniverse uses.
    """

    conn: Any

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

        # 1. AlphaUniverse (Plugins are auto-discovered by the Registry)
        self.lab = AlphaUniverse(conn=self.data_provider.conn, config=self.config)
        
        # SENIOR FIX: Use label_mode from config instead of hardcoded residual
        label_mode = self.config.get('signal_physics', {}).get('label_mode', 'directional')
        self.lab.labeler = AlphaLabeler(mode=label_mode)

        # 2. Ingestor
        self.ingestor = InstitutionalIngestor(self.data_provider, config=self.config)

        # 3. MetaController for belief scoring
        self.meta_controller = BayesianMetaController(
            prior_belief=self.config.get('risk_metacontroller', {}).get('prior_belief', 0.75),
        )

        # 4. Model specs for TFT
        lookback = self.config['signal_physics'].get('lookback_days', 63)
        n_scales = len(self.config['signal_physics']['wavelet_transform']['scales'])

        self.specs = {
            'static': {'x_static': 1},
            'past': {
                'x_seq': 1,
                'x_spatial': n_scales,
                'x_volume': 1,
                'x_momentum': 3,
                'x_calendar': 4,
            },
        }
        self.past_keys = sorted(self.specs['past'].keys())

        # 5. Load the trained "Sniper"
        model_path = self.config.get('model_pipeline', {}).get('model_path', 'models/sniper_v7.pt')
        hidden_dim = self.config.get('model_pipeline', {}).get('architecture', {}).get('hidden_dim', 32)
        trading_mode = self.config.get('execution_muscle', {}).get('trading_mode', 'sim')

        self.model = SniperRanker(specs=self.specs, hidden_dim=hidden_dim)

        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                logger.info(f"StrategyEngine: Loaded SniperRanker ({model_path}).")
            except Exception as e:
                msg = f"StrategyEngine: Model load failed ({e}). Check that `model_pipeline.architecture.hidden_dim` in config.yaml matches the trained checkpoint."
                if trading_mode != 'sim':
                    logger.error(f"FATAL: {msg}")
                    raise RuntimeError(msg)
                else:
                    logger.warning(f"{msg} Using uninitialized model for research/sim.")
        else:
            msg = f"StrategyEngine: Model file {model_path} not found."
            if trading_mode != 'sim':
                logger.error(f"FATAL: {msg}")
                raise RuntimeError(msg)
            else:
                logger.warning(f"{msg} Using uninitialized model for research/sim.")

        # Diagnostics cache: avoids running a full-universe inference for every
        # single-ticker UI drill-down. Invalidated whenever `as_of` changes.
        self._view_cache_as_of: Optional[datetime] = None
        self._view_cache: Optional[Dict[str, Any]] = None

    def ingest_data(self, tickers: List[str], start_date: str, end_date: str):
        """Dedicated method for manual ingestion."""
        if not self.ingestor:
            logger.warning("StrategyEngine: Ingestor not configured.")
            return
        self.ingestor.ingest_universe(tickers, start_date, end_date)

    def get_current_rankings(
        self,
        as_of: datetime,
        include_batch: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns a clean 'House View' of sorted tickers, scores, and signal energy.
        """
        tickers = self.config['universe']['tickers']

        lookback = self.config.get('signal_physics', {}).get('lookback_days', 63)
        batch = self.lab.snapshot(
            as_of=as_of, tickers=tickers, lookback=lookback, require_labels=False
        )

        if not batch:
            view = {
                "ladder": [],
                "belief_score": self.meta_controller.get_position_scaler(),
                "signal_energy": 0.0,
                "as_of": as_of.isoformat(),
                "status": "DATA_MISSING",
            }
            self._view_cache_as_of = as_of
            self._view_cache = view
            return view

        with torch.no_grad():
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in batch.data.items()}

            tft_inputs = {
                'static': {
                    k.replace('x_static_', ''): v
                    for k, v in inputs.items()
                    if k.startswith('x_static_')
                },
                'past': {
                    k.replace('x_past_', ''): v
                    for k, v in inputs.items()
                    if k.startswith('x_past_')
                },
            }

            out_dict = self.model.model(tft_inputs)

            # Median quantile (index 1) for ranking
            scores = out_dict['out'][:, 1].cpu().numpy()
            # Average VSN weights over the sequence length for interpretability
            past_weights = out_dict['past_weights'].mean(dim=1).cpu().numpy()

            # Vectorized signal energy (spatial wavelet mean magnitude)
            energy_all = torch.zeros(len(batch.tickers))
            if 'x_past_x_spatial' in batch.data:
                energy_all = torch.mean(
                    torch.abs(batch.data['x_past_x_spatial']).flatten(1), dim=1
                ).cpu()

        # Past keys used by the actual model (already excludes x_spatial in SniperTFT)
        model_past_keys = (
            self.model.model.past_keys
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'past_keys')
            else [k for k in self.past_keys if k != 'x_spatial']
        )

        ladder = []
        for i, ticker in enumerate(batch.tickers):
            score_val = float(scores[i]) if len(scores) > i else 0.0

            shap = {}
            for j, p_name in enumerate(model_past_keys):
                if j < past_weights.shape[1]:
                    shap[p_name] = float(past_weights[i, j])

            raw_price = 0.0
            if 'raw_price' in batch.data:
                raw_price = float(batch.data['raw_price'][i].item())

            energy = float(energy_all[i].item())

            ladder.append({
                "ticker": ticker,
                "score": score_val,
                "price": raw_price,
                "energy": energy,
                "shap": shap,
            })

        ladder.sort(key=lambda x: x['score'], reverse=True)

        global_energy = 0.0
        if 'x_past_x_spatial' in batch.data:
            global_energy = float(torch.mean(torch.abs(batch.data['x_past_x_spatial'])).item())

        view = {
            "ladder": ladder,
            "belief_score": self.meta_controller.get_position_scaler(),
            "signal_energy": global_energy,
            "as_of": as_of.isoformat(),
            "status": "OK",
        }

        # Cache for diagnostics drill-downs
        self._view_cache_as_of = as_of
        self._view_cache = view

        if include_batch:
            view["_batch"] = batch

        return view

    def get_ticker_diagnostics(
        self,
        ticker: str,
        as_of: datetime,
        house_view: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Provides deep spectral insights for a specific ticker.

        If `house_view` is provided, its SHAP weights are reused — avoids
        a full-universe re-inference. Otherwise the engine reuses its
        cached view from the same `as_of`, or computes a fresh one.
        """
        lookback = self.config.get('signal_physics', {}).get('lookback_days', 63)
        batch = self.lab.snapshot(
            as_of=as_of, tickers=[ticker], lookback=lookback, require_labels=False
        )
        if not batch:
            return None

        # 1. Extract History
        raw_prices = self.lab.get_batch_pit_view(
            [ticker], as_of, start_time=as_of - timedelta(days=lookback)
        )
        history_ui = []
        if not raw_prices.empty:
            for t, row in raw_prices.iterrows():
                history_ui.append({
                    "time": int(t.timestamp()),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": int(row['volume']),
                })

        # 2. Extract Wavelet CWT
        cwt = np.zeros((16, lookback))
        if 'x_past_x_spatial' in batch.data:
            cwt = batch.data['x_past_x_spatial'][0].cpu().numpy().T

        # 3. Real ADF test
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
                adf_p = float(torch.std(batch.data['x_past_x_seq']).item())

        # 4. SHAP — reuse caller-supplied or cached view; else recompute
        shap = {}
        view = house_view
        if view is None:
            if self._view_cache is not None and self._view_cache_as_of == as_of:
                view = self._view_cache
            else:
                view = self.get_current_rankings(as_of)

        for entry in view.get('ladder', []):
            if entry['ticker'] == ticker:
                shap = entry['shap']
                break

        return {
            "ticker": ticker,
            "history": history_ui,
            "cwt": cwt,
            "adf_p": adf_p,
            "shap_fusion": shap,
        }

    def update_model_metacognition(
        self,
        realized_returns: Dict[str, float],
        predicted_scores: Dict[str, float],
    ):
        """Updates the Bayesian MetaController."""
        tickers = list(realized_returns.keys())
        y_true = np.array([realized_returns[t] for t in tickers])
        y_pred = np.array([predicted_scores[t] for t in tickers])

        new_belief = self.meta_controller.update_belief(y_true, y_pred)
        return new_belief
