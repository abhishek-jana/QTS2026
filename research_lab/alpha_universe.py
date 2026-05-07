import pandas as pd
import numpy as np
import torch
import duckdb
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta, time
from qts_core.logger import logger
from joblib import Parallel, delayed

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
    def __init__(self, conn, plugins: List = None, config: dict = None):
        from research_lab.plugins.core_plugins import ModalityRegistry
        from research_lab.alpha_core import DiurnalStandardizer
        
        # Share the connection from DataEngine
        self._conn = conn
        self.labeler = None 
        self.config = config or {}
        self.db_path = self.config.get('data_engine', {}).get('storage_path')

        if plugins is not None: self.plugins = plugins
        else:
            self.plugins = ModalityRegistry.create_all(self.config)
            logger.info(f"AlphaUniverse: Auto-discovered plugins: {[p.name for p in self.plugins]}")
        
        self.timeframe = self.config.get('data_engine', {}).get('timeframe', '1Day')
        self.lookback = self.config.get('signal_physics', {}).get('lookback_days', 63)
        self.horizon = self.config.get('signal_physics', {}).get('horizon_days', 21)
        self.padding = self.config.get('signal_physics', {}).get('math_padding', 100)
        
        self.standardizer = DiurnalStandardizer() if 'Min' in self.timeframe else None
        # Cache for stationary features
        self._feat_cache = {}

    @property
    def conn(self):
        if self._conn is None:
            if self.db_path:
                import duckdb
                # SENIOR FIX: Open in read-only mode for workers to avoid locking issues
                # This ensures that parallel walk-forward doesn't crash on DB access
                self._conn = duckdb.connect(self.db_path, read_only=True)
            else:
                raise ValueError("DuckDB connection lost and no db_path available for re-init.")
        return self._conn

    def __getstate__(self):
        state = self.__dict__.copy()
        # SENIOR FIX: DuckDB connections cannot be pickled.
        # We null it out and let the @property re-establish it in the worker process.
        state['_conn'] = None 
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_batch_pit_view(self, tickers: List[str], as_of: datetime, start_time: Optional[datetime] = None) -> pd.DataFrame:
        ticker_list = "', '".join(tickers)
        standard_cols = ['ticker', 'event_time', 'knowledge_time', 'open', 'high', 'low', 'close', 'volume', 'is_correction']
        col_str = ", ".join(standard_cols)
        
        time_filter = f"AND knowledge_time <= '{as_of.isoformat()}'"
        if start_time:
            time_filter += f" AND event_time >= '{start_time.isoformat()}'"
            
        query = f"""
            WITH ranked_data AS (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY ticker, event_time ORDER BY knowledge_time DESC) as rn
                FROM market_data
                WHERE ticker IN ('{ticker_list}') {time_filter}
            )
            SELECT {col_str} FROM ranked_data WHERE rn = 1 ORDER BY event_time ASC
        """
        # SENIOR FIX: Use .df() for optimized transfer to pandas, avoids RecordBatchReader issues
        pit_view = self.conn.execute(query).df()
        if pit_view.empty: return pit_view
        return pit_view.set_index('event_time')

    def snapshot(self, as_of: datetime, tickers: List[str], lookback: int = None, horizon: int = None, latest_only: bool = True, backtest_mode: bool = False) -> Optional[MultiModalBatch]:
        from research_lab.alpha_labeler import AlphaLabeler
        if not self.labeler: self.labeler = AlphaLabeler()
        lookback, horizon = lookback or self.lookback, horizon or self.horizon
        total_window = lookback + self.padding
        
        if 'Min' in self.timeframe:
            days_back = max((total_window // 26) + 10, 60)
        else:
            days_back = max(total_window + 10, 100)
            
        fetch_start = as_of - timedelta(days=days_back)
        
        if 'Min' in self.timeframe:
            fetch_limit = as_of + timedelta(days=horizon + 10) if backtest_mode else as_of
            batch_view = self._get_batch_pit_view(tickers, fetch_limit, start_time=fetch_start)
        else:
            fetch_limit = as_of + timedelta(days=horizon * 2) if backtest_mode else as_of
            batch_view = self._get_batch_pit_view(tickers, fetch_limit, start_time=fetch_start)
            
        if batch_view.empty: return None
        
        processed_views = {}
        for ticker in tickers:
            raw_ticker_data = batch_view[batch_view['ticker'] == ticker]
            if raw_ticker_data.empty: continue
            if len(raw_ticker_data[raw_ticker_data.index <= as_of]) < lookback: continue
            
            std_ticker_data = raw_ticker_data.copy()
            if self.standardizer:
                std_ticker_data = self.standardizer.transform(std_ticker_data)
                
            processed_views[ticker] = {
                'raw': raw_ticker_data[raw_ticker_data.index <= fetch_limit],
                'std': std_ticker_data[std_ticker_data.index <= fetch_limit]
            }
            
        if not processed_views: return None
        
        # SENIOR OBSERVABILITY FIX: Periodic logging in worker to show progress
        # Since this runs in a worker, we log occasionally based on the day of the week
        if as_of.day % 7 == 0:
            logger.info(f"✨ Worker processing snapshot as of {as_of.date()}...")
            
        combined_raw_view = pd.concat([v['raw'] for v in processed_views.values()])
        returns_df = self.labeler.generate_labels(combined_raw_view, horizon=horizon, timeframe=self.timeframe)
        market_proxy = returns_df['SPY'] if 'SPY' in returns_df.columns else returns_df.mean(axis=1)
        residuals = self.labeler.residualize_universe(returns_df.drop(columns=['SPY']) if 'SPY' in returns_df.columns else returns_df, market_proxy)
        z_scored_labels = self.labeler.apply_z_score(residuals)
        
        fused_data = {p.name: [] for p in self.plugins}; fused_data['raw_price'] = []
        y_list, final_tickers, final_times = [], [], []
        
        for ticker, views in processed_views.items():
            std_view = views['std']
            raw_view = views['raw']
            
            feat_view = std_view[std_view.index <= as_of]
            if latest_only:
                if 'Min' in self.timeframe:
                    closes = feat_view[feat_view.index.time >= time(15, 45)]
                    if closes.empty: continue
                    feat_view = std_view[std_view.index <= closes.index[-1]].tail(total_window)
                else:
                    feat_view = feat_view.tail(total_window)
            
            # MEMOIZATION: Use cache
            cache_key = (ticker, feat_view.index[-1])
            if cache_key not in self._feat_cache:
                self._feat_cache[cache_key] = {p.name: p.transform(feat_view, lookback) for p in self.plugins}
            ticker_features = self._feat_cache[cache_key]
            
            ticker_labels = z_scored_labels[ticker] if ticker in z_scored_labels.columns else pd.Series(dtype=float)
            all_indices = range(lookback - 1, len(feat_view))
            if latest_only:
                indices = [all_indices[-1]] if len(all_indices) > 0 else []
            else:
                indices = [i for i in all_indices if feat_view.index[i].time() >= time(15, 45)]
            
            for i, t_idx in enumerate(all_indices):
                if t_idx not in indices: continue
                event_time = feat_view.index[t_idx]
                label_val = ticker_labels.get(event_time, np.nan)
                if backtest_mode and np.isnan(label_val): continue
                
                for p_name, tensor in ticker_features.items(): fused_data[p_name].append(tensor[i])
                fused_data['raw_price'].append(torch.tensor(float(raw_view.loc[event_time, 'close'])))
                y_list.append(label_val); final_tickers.append(ticker); final_times.append(event_time)
                
        if not final_tickers: return None
        return MultiModalBatch(data={name: torch.stack(tensors) for name, tensors in fused_data.items()}, labels=torch.tensor(np.array(y_list)).float(), tickers=final_tickers, times=final_times)

    def walk_forward(self, universe: List[str], start_date: datetime, end_date: datetime, stride: int = 21, **kwargs):
        results = []
        def _get_dates():
            current = start_date
            while current <= end_date:
                yield current
                current += timedelta(days=stride)
        dates = list(_get_dates())
        
        logger.info(f"🚀 Parallelizing walk-forward over {len(dates)} days...")
        
        # SENIOR FIX: Close the main connection before parallelizing to avoid DuckDB lock contention.
        # Worker processes will re-open it in read-only mode.
        if self._conn:
            try:
                self._conn.close()
            except Exception: pass
            self._conn = None

        walk_latest_only = kwargs.pop('latest_only', True)
        walk_backtest_mode = kwargs.pop('backtest_mode', True)
        
        batch_results = Parallel(n_jobs=-1)(
            delayed(self.snapshot)(
                as_of=d, tickers=universe, latest_only=walk_latest_only, backtest_mode=walk_backtest_mode, **kwargs
            ) for d in dates
        )
        
        for i, batch in enumerate(batch_results):
            if batch: results.append({'date': dates[i], 'batch': batch})
            
        return results
