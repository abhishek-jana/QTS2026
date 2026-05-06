import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from qts_core.logger import logger

@dataclass
class MultiModalBatch:
    data: Dict[str, torch.Tensor]
    labels: torch.Tensor
    tickers: List[str]
    times: List[datetime]
    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        if isinstance(index, str): return self.data[index]
        item = {name: tensor[index] for name, tensor in self.data.items()}
        item['y'] = self.labels[index]
        return item
    def to(self, device: torch.device):
        self.data = {k: v.to(device) for k, v in self.data.items()}
        self.labels = self.labels.to(device)
        return self

class AlphaUniverse:
    def __init__(self, data_provider, plugins: List = None, config: dict = None):
        from research_lab.plugins.core_plugins import ModalityRegistry
        self.data_provider = data_provider
        self.labeler = None 
        self.config = config or {}
        if plugins is not None: self.plugins = plugins
        else:
            self.plugins = ModalityRegistry.create_all(self.config.get('plugins', {}))
            logger.info(f"AlphaUniverse: Auto-discovered plugins: {[p.name for p in self.plugins]}")
        self.lookback = self.config.get('signal_physics', {}).get('lookback_days', 63)
        self.horizon = self.config.get('signal_physics', {}).get('horizon_days', 21)
        self.padding = self.config.get('signal_physics', {}).get('math_padding', 100)

    def snapshot(self, as_of: datetime, tickers: List[str], lookback: int = None, horizon: int = None, latest_only: bool = True, backtest_mode: bool = False) -> Optional[MultiModalBatch]:
        from research_lab.alpha_labeler import AlphaLabeler
        if not self.labeler: self.labeler = AlphaLabeler()
        lookback, horizon = lookback or self.lookback, horizon or self.horizon
        total_window = lookback + self.padding
        fetch_limit = as_of + timedelta(days=horizon * 2) if backtest_mode else as_of
        batch_view = self.data_provider.get_batch_pit_view(tickers, fetch_limit)
        if batch_view.empty: return None
        processed_views = {}
        for ticker in tickers:
            ticker_data = batch_view[batch_view['ticker'] == ticker]
            if ticker_data.empty: continue
            feat_view = ticker_data[ticker_data.index <= as_of]
            if len(feat_view) < lookback: continue
            processed_views[ticker] = ticker_data[ticker_data.index <= fetch_limit]
        if not processed_views: return None
        combined_view = pd.concat(processed_views.values())
        returns_df = self.labeler.generate_labels(combined_view, horizon=horizon)
        market_proxy = returns_df['SPY'] if 'SPY' in returns_df.columns else returns_df.mean(axis=1)
        residuals = self.labeler.residualize_universe(returns_df.drop(columns=['SPY']) if 'SPY' in returns_df.columns else returns_df, market_proxy)
        z_scored_labels = self.labeler.apply_z_score(residuals)
        fused_data = {p.name: [] for p in self.plugins}; fused_data['raw_price'] = []
        y_list, final_tickers, final_times = [], [], []
        for ticker, full_view in processed_views.items():
            feat_view = full_view[full_view.index <= as_of]
            if latest_only: feat_view = feat_view.tail(total_window)
            ticker_labels = z_scored_labels[ticker] if ticker in z_scored_labels.columns else pd.Series(dtype=float)
            ticker_features = {p.name: p.transform(feat_view, lookback) for p in self.plugins}
            indices = range(lookback - 1, len(feat_view))
            if latest_only and len(indices) > 0: indices = [indices[-1]]
            for i, t_idx in enumerate(range(lookback - 1, len(feat_view))):
                if latest_only and t_idx != indices[0]: continue
                event_time = feat_view.index[t_idx]
                label_val = ticker_labels.get(event_time, np.nan)
                if backtest_mode and np.isnan(label_val): continue
                for p_name, tensor in ticker_features.items(): fused_data[p_name].append(tensor[i])
                fused_data['raw_price'].append(torch.tensor(float(feat_view['close'].iloc[t_idx])))
                y_list.append(label_val); final_tickers.append(ticker); final_times.append(event_time)
        if not final_tickers: return None
        return MultiModalBatch(data={name: torch.stack(tensors) for name, tensors in fused_data.items()}, labels=torch.tensor(np.array(y_list)).float(), tickers=final_tickers, times=final_times)

    def walk_forward(self, universe: List[str], start_date: datetime, end_date: datetime, stride: int = 21, **kwargs):
        results = []
        current_date = start_date
        # SENIOR FIX: Properly handle argument priority
        walk_latest_only = kwargs.pop('latest_only', True)
        walk_backtest_mode = kwargs.pop('backtest_mode', True)
        while current_date <= end_date:
            logger.info(f"Executing Lab Walk-Forward: {current_date.date()}")
            batch = self.snapshot(as_of=current_date, tickers=universe, latest_only=walk_latest_only, backtest_mode=walk_backtest_mode, **kwargs)
            if batch: results.append({'date': current_date, 'batch': batch})
            current_date += timedelta(days=stride)
        return results
