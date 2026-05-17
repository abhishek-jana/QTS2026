import logging
from datetime import datetime, timedelta

import pandas as pd
import torch

from research_lab.data_engine import DataEngine
from research_lab.alpha_ranker import RankNet

logger = logging.getLogger(__name__)


class WFOEngine:
    """
    Walk-Forward Optimization Engine.
    Automates model retraining while strictly honoring Knowledge Time.

    NOTE: This engine currently wraps the legacy `RankNet`. The production
    Sniper-Residual strategy uses `SniperRanker` (see
    `research_lab/alpha_ranker_sniper.py`). A migration to SniperRanker
    is planned; until then this WFO engine is used only for the legacy
    RankNet pipeline and integration smoke tests.
    """

    def __init__(self, data_engine: DataEngine, model_params: dict):
        self.data_engine = data_engine
        self.model_params = model_params
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"WFO Engine initialized on device: {self.device}")

    def train_step(self, train_view: pd.DataFrame, target_labels: pd.DataFrame):
        """
        Trains a single RankNet instance on the provided PIT view.
        Uses the high-level .fit() interface with GPU support.
        """
        if train_view is None or len(train_view) == 0:
            raise ValueError("train_step: train_view is empty.")
        if target_labels is None or len(target_labels) == 0:
            raise ValueError("train_step: target_labels are empty.")
        if len(train_view) != len(target_labels):
            raise ValueError(
                f"train_step: feature/label length mismatch "
                f"({len(train_view)} vs {len(target_labels)})."
            )

        input_dim = self.model_params['input_dim']
        model = RankNet(input_dim).to(self.device)

        X = torch.tensor(train_view.values).float().to(self.device)
        y = torch.tensor(target_labels.values).float().to(self.device)

        model.fit(
            (X, y),
            epochs=self.model_params.get('epochs', 50),
            device=self.device,
        )
        return model.state_dict()

    def _build_labels(self, train_view: pd.DataFrame) -> pd.DataFrame:
        """
        Build forward-return labels from the PIT view. Subclasses or callers
        may override; the default uses a `horizon_days` ahead simple return
        from the `close` column.
        """
        if 'close' not in train_view.columns:
            raise NotImplementedError(
                "WFOEngine._build_labels: no `close` column in train_view. "
                "Either pass labels explicitly to `run_pipeline` or extend "
                "this engine with a domain-specific labeler."
            )
        horizon = self.model_params.get('horizon_days', 5)
        fwd = train_view['close'].shift(-horizon) / train_view['close'] - 1.0
        labels = fwd.dropna().to_frame('y')
        # Drop the trailing rows that have no forward return
        return labels

    def run_pipeline(
        self,
        start_date: datetime,
        end_date: datetime,
        step_days: int = 30,
        label_builder=None,
    ):
        """
        Iterates through time, retraining the model at each step.

        label_builder: optional callable(train_view) -> labels DataFrame.
                       If None, uses `self._build_labels` (forward-return).
        """
        current_date = start_date
        builder = label_builder or self._build_labels

        while current_date <= end_date:
            train_view = self.data_engine.registry[
                self.data_engine.registry['knowledge_time'] <= current_date
            ]

            if len(train_view) < self.model_params.get('min_train_samples', 100):
                logger.warning(
                    f"WFO skip @ {current_date}: only {len(train_view)} "
                    f"samples available (need "
                    f"{self.model_params.get('min_train_samples', 100)})."
                )
                current_date += timedelta(days=step_days)
                continue

            try:
                labels = builder(train_view)
                feats = train_view.loc[labels.index]
                state = self.train_step(feats, labels)
                self.models[current_date] = state
                logger.info(
                    f"WFO retrain @ {current_date} | "
                    f"samples={len(feats)} | checkpoints stored={len(self.models)}"
                )
            except Exception as e:
                logger.error(f"WFO retrain @ {current_date} failed: {e}")

            current_date += timedelta(days=step_days)

        return self.models
