import pandas as pd
import numpy as np
import torch
import duckdb
import gc
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta, time
from qts_core.logger import logger
from tqdm import tqdm

@dataclass
class MultiModalBatch:
    data: Dict[str, torch.Tensor]
    labels: torch.Tensor
    tickers: List[str]
    times: List[datetime]
    sample_weights: Optional[torch.Tensor] = None
    
    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        if isinstance(index, str): return self.data[index]
        item = {name: tensor[index] for name, tensor in self.data.items()}
        item['y'] = self.labels[index]
        if self.sample_weights is not None:
            item['w'] = self.sample_weights[index]
        return item
    def to(self, device: torch.device):
        self.data = {k: v.to(device) for k, v in self.data.items()}
        self.labels = self.labels.to(device)
        if self.sample_weights is not None:
            self.sample_weights = self.sample_weights.to(device)
        return self

class AlphaUniverse:
    def __init__(self, conn, plugins: List = None, config: dict = None):
        from research_lab.plugins.core_plugins import ModalityRegistry
        from research_lab.alpha_core import DiurnalStandardizer
        
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
        self._feat_cache = None

    @property
    def conn(self):
        is_closed = True
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")
                is_closed = False
            except Exception:
                is_closed = True
        
        if is_closed:
            if self.db_path:
                import duckdb
                self._conn = duckdb.connect(self.db_path, read_only=True)
            else:
                raise ValueError("DuckDB connection lost.")
        return self._conn

    def get_batch_pit_view(self, tickers: List[str], as_of: datetime, start_time: Optional[datetime] = None) -> pd.DataFrame:
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
        pit_view = self.conn.execute(query).df()
        if pit_view.empty: return pit_view
        return pit_view.set_index('event_time')

    def snapshot(self, as_of: datetime, tickers: List[str], lookback: int = None, labels_override: pd.DataFrame = None, shard_override: Dict[str, pd.DataFrame] = None) -> Optional[MultiModalBatch]:
        """
        SENIOR V5.8.6: Ultra-Fast Shard-Based Snapshot.
        Replaces boolean masking with O(1) shard lookups.
        """
        lookback = lookback or self.lookback
        
        # SENIOR FIX: Support standalone calls (e.g. for doctor.py or UI)
        if shard_override is None:
            total_window = lookback + self.padding
            days_back = max(total_window + 10, 100)
            fetch_start = as_of - timedelta(days=days_back)
            batch_view = self.get_batch_pit_view(tickers, as_of, start_time=fetch_start)
            if batch_view.empty: return None
            # Temporary sharding
            shard_override = {t: batch_view[batch_view['ticker'] == t] for t in tickers}
            
        final_labels_df = labels_override
        fused_data = {p.name: [] for p in self.plugins}; fused_data['raw_price'] = []
        y_list, final_tickers, final_times = [], [], []
        
        for ticker in tickers:
            if ticker not in shard_override: continue
            full_ticker_data = shard_override[ticker]
            
            ticker_slice_all = full_ticker_data[full_ticker_data.index <= as_of]
            if len(ticker_slice_all) < lookback: continue
            
            actual_as_of = ticker_slice_all.index[-1]
            ticker_slice = ticker_slice_all.tail(lookback + 5)
            
            ticker_features = {p.name: p.transform(ticker_slice, lookback) for p in self.plugins}
            
            if final_labels_df is None:
                # Local label computation for standalone calls
                from research_lab.alpha_labeler import AlphaLabeler
                if not self.labeler: self.labeler = AlphaLabeler()
                # Use a small window for local labels
                label_limit = as_of + timedelta(days=self.horizon * 2)
                local_data = self.get_batch_pit_view([ticker, 'SPY'], label_limit, start_time=as_of - timedelta(days=5))
                if local_data.empty: continue
                local_returns = self.labeler.generate_labels(local_data, horizon=self.horizon, timeframe=self.timeframe)
                if 'SPY' in local_returns.columns: excess = local_returns[ticker] - local_returns['SPY']
                else: excess = local_returns[ticker]
                
                # SENIOR V6.3 FIX: Rectified Local Labels
                label_val = max(0.0, float(excess.get(actual_as_of, np.nan)))
            else:
                ticker_labels = final_labels_df[ticker] if ticker in final_labels_df.columns else pd.Series(dtype=float)
                label_val = ticker_labels.get(actual_as_of, np.nan)
            
            if np.isnan(label_val): continue
            
            for p_name, tensor in ticker_features.items(): 
                fused_data[p_name].append(tensor[-1])
                
            fused_data['raw_price'].append(torch.tensor(float(ticker_slice.loc[actual_as_of, 'close'])))
            y_list.append(label_val); final_tickers.append(ticker); final_times.append(actual_as_of)
                
        if not final_tickers: return None
        return MultiModalBatch(data={name: torch.stack(tensors) for name, tensors in fused_data.items()}, labels=torch.tensor(np.array(y_list)).float(), tickers=final_tickers, times=final_times)

    def walk_forward(self, universe: List[str], start_date: datetime, end_date: datetime, stride: int = 1, **kwargs):
        """
        SENIOR V5.8.6: Sharded-Memory Walk-Forward.
        Fixed time-alignment logic to process 15-min data with daily strides.
        """
        results = []
        
        from research_lab.alpha_labeler import AlphaLabeler
        if not self.labeler: self.labeler = AlphaLabeler()
        
        logger.info(f"📈 Initiating Shard-Based Pre-computation for {len(universe)} tickers...")
        total_window = self.lookback + self.padding
        data_fetch_limit = end_date + timedelta(days=self.horizon * 2)
        fetch_start = start_date - timedelta(days=total_window + 30)
        
        all_history = self.get_batch_pit_view(universe, data_fetch_limit, start_time=fetch_start)
        if all_history.empty: return []
            
        raw_returns = self.labeler.generate_labels(all_history, horizon=self.horizon, timeframe=self.timeframe)
        if 'SPY' in raw_returns.columns: excess = raw_returns.sub(raw_returns['SPY'], axis=0)
        else: excess = raw_returns.sub(raw_returns.mean(axis=1), axis=0)
        
        # SENIOR V6.3 FIX: Rectified "Growth Hunter" Targets
        # Force the model to only care about positive alpha. Negative alpha is noise.
        excess = excess.clip(lower=0.0)
        self.precomputed_labels = self.labeler.apply_z_score(excess)
        
        shards = {t: all_history[all_history['ticker'] == t].copy() for t in universe}
        del all_history 
        gc.collect()

        dates = []
        curr = start_date.replace(hour=16, minute=0, second=0, microsecond=0)
        while curr <= end_date.replace(hour=16, minute=0, second=0, microsecond=0):
            if curr.weekday() < 5:
                dates.append(curr)
            curr += timedelta(days=stride)
            
        logger.info(f"🚀 Sharded Loop starting. Total tasks: {len(dates)}")

        for d in tqdm(dates, desc="🏗️ Building Dataset"):
            batch = self.snapshot(as_of=d, tickers=universe, lookback=self.lookback, 
                                 labels_override=self.precomputed_labels,
                                 shard_override=shards)
            if batch:
                results.append({'date': d, 'batch': batch})
            
            if len(results) > 0 and len(results) % 100 == 0: gc.collect()
                
        logger.success(f"✅ Data Extraction Complete. Total Samples: {len(results)}")
        return results
