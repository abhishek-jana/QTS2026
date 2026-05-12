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
        
        self.timeframe = self.config.get('data_engine', {}).get('timeframe', '15Min')
        self.lookback = self.config.get('signal_physics', {}).get('lookback_days', 63)
        self.horizon = self.config.get('signal_physics', {}).get('horizon_days', 3) # Sniper-Residual default
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
        Sniper V7.0: TFT-Ready Sequence Snapshot.
        Distinguishes between static and past (temporal) inputs.
        """
        lookback = lookback or self.lookback
        
        if shard_override is None:
            total_window = lookback + self.padding
            days_back = max(total_window + 10, 100)
            fetch_start = as_of - timedelta(days=days_back)
            batch_view = self.get_batch_pit_view(tickers, as_of, start_time=fetch_start)
            if batch_view.empty: return None
            shard_override = {t: batch_view[batch_view['ticker'] == t] for t in tickers}
            
        final_labels_df = labels_override
        fused_data = {}
        y_list, final_tickers, final_times = [], [], []
        
        for ticker in tickers:
            if ticker not in shard_override: continue
            full_ticker_data = shard_override[ticker]
            
            ticker_slice_all = full_ticker_data[full_ticker_data.index <= as_of]
            if len(ticker_slice_all) < lookback: continue
            
            actual_as_of = ticker_slice_all.index[-1]
            ticker_slice = ticker_slice_all.tail(lookback + self.padding)
            
            ticker_features = {p.name: p.transform(ticker_slice, lookback) for p in self.plugins}
            
            if final_labels_df is None:
                # Local computation if no override provided (for testing)
                from research_lab.alpha_labeler import AlphaLabeler
                if not self.labeler: self.labeler = AlphaLabeler()
                # Increase label_limit to handle weekends/holidays (Sniper V7.4.3 Fix)
                label_limit = as_of + timedelta(days=self.horizon + 12)
                local_data = self.get_batch_pit_view([ticker, 'SPY'], label_limit, start_time=as_of - timedelta(days=1))
                if local_data.empty: continue
                local_returns = self.labeler.generate_labels(local_data, horizon_days=self.horizon, timeframe=self.timeframe)
                label_val = float(local_returns[ticker].get(actual_as_of, np.nan))
            else:
                ticker_labels = final_labels_df[ticker] if ticker in final_labels_df.columns else pd.Series(dtype=float)
                label_val = ticker_labels.get(actual_as_of, np.nan)
            
            if np.isnan(label_val): continue
            
            for p_name, tensor in ticker_features.items(): 
                # TFT Distinction: 'x_static' is [dim], 'x_past' is [seq, dim]
                if p_name == 'x_static':
                    f_key = f"x_static_{p_name}"
                    if f_key not in fused_data: fused_data[f_key] = []
                    fused_data[f_key].append(tensor[-1]) # [dim]
                else:
                    f_key = f"x_past_{p_name}"
                    if f_key not in fused_data: fused_data[f_key] = []
                    fused_data[f_key].append(tensor[-1]) # [lookback, dim]
                
            y_list.append(label_val); final_tickers.append(ticker); final_times.append(actual_as_of)
                
        if not final_tickers: return None
        return MultiModalBatch(
            data={name: torch.stack(tensors) for name, tensors in fused_data.items()}, 
            labels=torch.tensor(np.array(y_list)).float(), 
            tickers=final_tickers, 
            times=final_times
        )

    def walk_forward(self, universe: List[str], start_date: datetime, end_date: datetime, stride: int = 1, **kwargs):
        """
        Sniper V7.0: ULTRA-FAST Vectorized Residual Walk-Forward.
        Pre-computes all features before building daily batches to eliminate 
        O(Days * Tickers) loop bottlenecks. Includes robust disk caching.
        """
        import os
        import hashlib
        
        # Create a deterministic cache key based on universe size and dates
        cache_key = f"wf_{len(universe)}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{self.horizon}_{stride}"
        cache_path = f"data/{cache_key}.pt"
        
        if os.path.exists(cache_path):
            logger.info(f"📦 Loading pre-computed Walk-Forward cache from {cache_path}...")
            try:
                # Add MultiModalBatch and pandas classes to safe globals
                from datetime import datetime as dt_class
                torch.serialization.add_safe_globals([MultiModalBatch, dt_class, pd.Timestamp, pd._libs.tslibs.timestamps._unpickle_timestamp])
                return torch.load(cache_path, weights_only=True)
            except Exception as e:
                logger.warning(f"Failed to load cache ({e}). Rebuilding...")
        
        results = []
        from research_lab.alpha_labeler import AlphaLabeler
        if not self.labeler: self.labeler = AlphaLabeler()
        
        logger.info(f"🏗️ Sniper Engine: Pre-fetching data for {len(universe)} tickers...")
        total_window = self.lookback + self.padding
        data_fetch_limit = end_date + timedelta(days=self.horizon + 5)
        fetch_start = start_date - timedelta(days=total_window + 30)
        
        all_history = self.get_batch_pit_view(universe, data_fetch_limit, start_time=fetch_start)
        if all_history.empty: return []
            
        # Target: 3-day Residual Log-Returns
        self.precomputed_labels = self.labeler.generate_labels(all_history, horizon_days=self.horizon, timeframe=self.timeframe)
        
        # PRE-COMPUTE ALL MODALITIES AT ONCE PER TICKER
        logger.info(f"⚡ Vectorizing plugin transforms across full time-series...")
        precomputed_features = {}
        precomputed_prices = {}
        for ticker in tqdm(universe, desc="Pre-computing Features"):
            ticker_data = all_history[all_history['ticker'] == ticker]
            if len(ticker_data) < self.lookback: continue
            
            # Store raw prices for sim engine
            valid_dates = ticker_data.index[self.lookback-1:]
            precomputed_prices[ticker] = {date: ticker_data.loc[date, 'close'] for date in valid_dates}
            
            ticker_features = {}
            for p in self.plugins:
                try:
                    tensor_data = p.transform(ticker_data, self.lookback)
                    if len(tensor_data) == len(valid_dates):
                        ticker_features[p.name] = {date: tensor_data[i] for i, date in enumerate(valid_dates)}
                except Exception as e:
                    logger.debug(f"Plugin {p.name} failed for {ticker}: {e}")
            precomputed_features[ticker] = ticker_features
            
        del all_history; gc.collect()

        # Generate walk-forward dates (Daily at 16:00 EST)
        dates = []
        curr = start_date.replace(hour=16, minute=0, second=0, microsecond=0)
        while curr <= end_date.replace(hour=16, minute=0, second=0, microsecond=0):
            if curr.weekday() < 5: dates.append(curr)
            curr += timedelta(days=stride)
            
        logger.info("📦 Assembling Multi-Modal Batches...")
        for d in tqdm(dates, desc="🏗️ Building Sniper Batches"):
            fused_data = {}
            y_list, final_tickers, final_times = [], [], []
            
            for ticker in universe:
                if ticker not in precomputed_features: continue
                ticker_feats = precomputed_features[ticker]
                
                # We need features exactly 'as_of' this date, or the closest prior valid date
                if d in ticker_feats.get('x_static', {}):
                    actual_as_of = d
                else:
                    # Find closest prior
                    valid_dates = [dt for dt in ticker_feats.get('x_static', {}).keys() if dt <= d]
                    if not valid_dates: continue
                    actual_as_of = max(valid_dates)
                    
                    if (d - actual_as_of).days > 3: continue
                
                ticker_labels = self.precomputed_labels[ticker] if ticker in self.precomputed_labels.columns else pd.Series(dtype=float)
                label_val = ticker_labels.get(actual_as_of, np.nan)
                if np.isnan(label_val): continue
                
                valid_sample = True
                for p_name in [p.name for p in self.plugins]:
                    if p_name not in ticker_feats or actual_as_of not in ticker_feats[p_name]:
                        valid_sample = False; break
                if not valid_sample: continue

                # Add raw price for sim engine
                if 'raw_price' not in fused_data: fused_data['raw_price'] = []
                fused_data['raw_price'].append(torch.tensor(float(precomputed_prices[ticker][actual_as_of])))

                for p_name in [p.name for p in self.plugins]:
                    tensor = ticker_feats[p_name][actual_as_of]
                    if p_name == 'x_static':
                        f_key = f"x_static_{p_name}"
                        if f_key not in fused_data: fused_data[f_key] = []
                        fused_data[f_key].append(tensor)
                    else:
                        f_key = f"x_past_{p_name}"
                        if f_key not in fused_data: fused_data[f_key] = []
                        fused_data[f_key].append(tensor)
                
                y_list.append(label_val)
                final_tickers.append(ticker)
                final_times.append(actual_as_of)
                
            if final_tickers:
                batch = MultiModalBatch(
                    data={name: torch.stack(tensors) for name, tensors in fused_data.items()}, 
                    labels=torch.tensor(np.array(y_list)).float(), 
                    tickers=final_tickers, 
                    times=final_times
                )
                results.append({'date': d, 'batch': batch})
                
        logger.success(f"✅ Fast Sniper Dataset Built: {len(results)} days.")
        
        try:
            os.makedirs('data', exist_ok=True)
            torch.save(results, cache_path)
            logger.info(f"💾 Walk-Forward cache saved to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save Walk-Forward cache: {e}")
            
        return results
