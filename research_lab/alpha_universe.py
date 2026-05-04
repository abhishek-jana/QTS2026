import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from research_lab.data_engine import DataEngine
from research_lab.alpha_labeler import AlphaLabeler
from research_lab.plugins.core_plugins import ModalityPlugin

@dataclass
class MultiModalBatch:
    """Container for fused data modalities and labels."""
    data: Dict[str, torch.Tensor]
    labels: torch.Tensor
    tickers: List[str]
    times: List[datetime]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        # Integer indexing for DataLoader compatibility
        return {
            'x_seq': self.data['x_seq'][index],
            'x_spatial': self.data['x_spatial'][index],
            'y': self.labels[index]
        }

class AlphaUniverse:
    """
    The Alpha Orchestrator: Synthesizes Snapshot and Walk-Forward designs.
    Deep Module for bi-temporally aligned multi-modal data.
    """
    def __init__(self, engine: DataEngine = None, plugins: List[ModalityPlugin] = None):
        self.engine = engine or DataEngine()
        self.labeler = AlphaLabeler()
        self.plugins = plugins or []

    def snapshot(self, as_of: datetime, tickers: List[str], lookback: int = 63, horizon: int = 21) -> Optional[MultiModalBatch]:
        """
        [MINIMALIST] The primary interface for research and notebooks.
        One call returns a perfectly aligned batch of all registered modalities.
        """
        all_pit_views = {t: self.engine.get_pit_view(t, as_of) for t in tickers}
        
        # 1. Generate target labels (Global Universe context)
        combined_view = pd.concat(all_pit_views.values())
        returns_df = self.labeler.generate_labels(combined_view, horizon=horizon)
        
        market_proxy = returns_df['SPY'] if 'SPY' in returns_df.columns else returns_df.mean(axis=1)
        residuals = self.labeler.residualize_universe(returns_df.drop(columns=['SPY']) if 'SPY' in returns_df.columns else returns_df, market_proxy)
        z_scored_labels = self.labeler.apply_z_score(residuals)

        # 2. Process Modalities per ticker
        fused_data = {p.name: [] for p in self.plugins}
        y_list = []
        final_tickers = []
        final_times = []

        for ticker, view in all_pit_views.items():
            if len(view) < lookback: continue
            if ticker not in z_scored_labels.columns: continue
            
            # Apply all plugins to this ticker's view
            ticker_features = {p.name: p.transform(view, lookback) for p in self.plugins}
            ticker_labels = z_scored_labels[ticker].dropna()
            
            # Align: Plugins return (N, ...) where N = len(view) - lookback + 1
            # These correspond to event_times from index [lookback-1:]
            for i, t_idx in enumerate(range(lookback - 1, len(view))):
                event_time = view.index[t_idx]
                
                if event_time not in ticker_labels.index: continue
                
                for p_name, tensor in ticker_features.items():
                    fused_data[p_name].append(tensor[i])
                
                y_list.append(ticker_labels.loc[event_time])
                final_tickers.append(ticker)
                final_times.append(event_time)

        if not y_list: return None

        return MultiModalBatch(
            data={name: torch.stack(tensors) for name, tensors in fused_data.items()},
            labels=torch.tensor(np.array(y_list)).float(),
            tickers=final_tickers,
            times=final_times
        )

    def walk_forward(self, universe: List[str], start_date: datetime, end_date: datetime, stride: int = 21, **kwargs):
        """
        [COMMON-CASE] Automated retraining/backtesting orchestrator.
        Steps through history using PIT-consistent snapshots.
        """
        results = []
        current_date = start_date
        while current_date <= end_date:
            print(f"Executing Lab Walk-Forward: {current_date}")
            batch = self.snapshot(as_of=current_date, tickers=universe, **kwargs)
            if batch:
                results.append({'date': current_date, 'batch': batch})
            current_date += timedelta(days=stride)
        return results
