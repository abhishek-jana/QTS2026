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

    def snapshot(self, as_of: datetime, tickers: List[str], lookback: int = None, labels_override: pd.DataFrame = None, shard_override: Dict[str, pd.DataFrame] = None, require_labels: bool = True) -> Optional[MultiModalBatch]:
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

            # SANITIZATION: Fill data gaps per ticker
            fill_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in fill_cols:
                if col in batch_view.columns:
                    batch_view[col] = batch_view.groupby('ticker')[col].ffill().bfill()

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
            
            label_val = 0.0
            if require_labels:
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
            
        # SANITIZATION: Fill data gaps per ticker (Forward-Fill then Backward-Fill)
        # This prevents NaNs from breaking wavelet transforms and signal physics.
        logger.info(f"🛠️ AlphaUniverse: Sanitizing {len(all_history)} rows of historical data...")
        fill_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in fill_cols:
            if col in all_history.columns:
                all_history[col] = all_history.groupby('ticker')[col].ffill().bfill()

        # Target: 3-day Residual Log-Returns
        self.precomputed_labels = self.labeler.generate_labels(all_history, horizon_days=self.horizon, timeframe=self.timeframe)
        
        # PRE-COMPUTE ALL MODALITIES AT ONCE PER TICKER
        # EFFICIENCY: store (sorted_ns_array, stacked_tensor) instead of
        # {date: tensor} dicts. This eliminates 540K Python hash lookups and
        # O(N) linear fallback scans during batch assembly. Alignment is done
        # once per ticker with a single np.searchsorted over all walk-forward
        # dates, replacing the per-day, per-ticker date resolution loop.
        logger.info(f"⚡ Vectorizing plugin transforms across full time-series...")

        def _ts_ns(ts) -> np.int64:
            """pd.Timestamp or datetime → int64 nanoseconds."""
            return np.int64(pd.Timestamp(ts).value)

        precomputed_features = {}  # {ticker: {p_name: (sorted_ns_int64, stacked_tensor)}}
        precomputed_prices   = {}  # {ticker: (sorted_ns_int64, close_float32)}

        for ticker in tqdm(universe, desc="Pre-computing Features"):
            ticker_data = all_history[all_history['ticker'] == ticker]
            if len(ticker_data) < self.lookback:
                continue

            valid_dates = ticker_data.index[self.lookback - 1:]          # pd.DatetimeIndex
            valid_ns    = np.array([_ts_ns(d) for d in valid_dates], dtype=np.int64)

            # Raw prices: numpy array aligned to valid_dates
            close_vals = ticker_data.loc[valid_dates, 'close'].values.astype(np.float32)
            precomputed_prices[ticker] = (valid_ns, close_vals)

            ticker_features = {}
            for p in self.plugins:
                try:
                    tensor_data = p.transform(ticker_data, self.lookback)  # (N, *shape)
                    if len(tensor_data) == len(valid_dates):
                        # Keep the full stacked tensor; don't build a dict of slices.
                        ticker_features[p.name] = (valid_ns, tensor_data)
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

        logger.info("📦 Pre-aligning tickers to walk-forward dates...")
        # EFFICIENCY: one np.searchsorted call per ticker covers all D dates at once.
        # This replaces 540K Python dict lookups and O(N) linear fallback scans
        # with O(T * log D) vectorised alignment, then O(1) tensor indexing per
        # (day, ticker, plugin) during assembly.

        wf_ns = np.array([pd.Timestamp(d).value for d in dates], dtype=np.int64)
        three_day_ns = np.int64(3 * 24 * 3600 * int(1e9))
        plugin_names = [p.name for p in self.plugins]          # built once
        p_key_map = {
            p.name: (f"x_static_{p.name}" if p.name == "x_static" else f"x_past_{p.name}")
            for p in self.plugins
        }

        # Pre-align labels: {ticker: (sorted_ns_int64, values_float32)}
        aligned_labels: dict = {}
        for col in self.precomputed_labels.columns:
            s = self.precomputed_labels[col].dropna()
            if s.empty:
                continue
            lns = np.array([pd.Timestamp(d).value for d in s.index], dtype=np.int64)
            aligned_labels[col] = (lns, s.values.astype(np.float32))

        # Pre-compute per-ticker alignment: one searchsorted over all D dates.
        ticker_info: dict = {}
        for ticker in universe:
            t_feats = precomputed_features.get(ticker)
            if not t_feats:
                continue
            ref_p = "x_static" if "x_static" in t_feats else next(iter(t_feats), None)
            if ref_p is None:
                continue
            ref_ns, _ = t_feats[ref_p]

            idxs      = np.searchsorted(ref_ns, wf_ns, side="right") - 1   # (D,)
            safe_idxs = np.maximum(idxs, 0)
            date_diff  = wf_ns - ref_ns[safe_idxs]
            date_valid = (idxs >= 0) & (date_diff <= three_day_ns)
            actual_ns  = np.where(date_valid, ref_ns[safe_idxs], np.int64(0))

            label_out   = np.full(len(dates), np.nan, dtype=np.float32)
            label_valid = np.zeros(len(dates), dtype=bool)
            if ticker in aligned_labels:
                lbl_ns, lbl_vals = aligned_labels[ticker]
                for d_i in np.where(date_valid)[0]:
                    li = int(np.searchsorted(lbl_ns, actual_ns[d_i], side="left"))
                    if li < len(lbl_ns) and lbl_ns[li] == actual_ns[d_i]:
                        v = lbl_vals[li]
                        if not np.isnan(v):
                            label_out[d_i]   = v
                            label_valid[d_i] = True

            plugin_ok = date_valid.copy()
            p_idx_map: dict = {}
            for p_name in plugin_names:
                if p_name not in t_feats:
                    plugin_ok[:] = False
                    break
                p_ns, _ = t_feats[p_name]
                if np.array_equal(p_ns, ref_ns):
                    p_idx_map[p_name] = safe_idxs
                else:
                    pi = np.searchsorted(p_ns, actual_ns, side="left")
                    pi = np.minimum(pi, len(p_ns) - 1)
                    plugin_ok[p_ns[pi] != actual_ns] = False
                    p_idx_map[p_name] = pi

            ticker_info[ticker] = {
                "feat_idx":  safe_idxs,
                "actual_ns": actual_ns,
                "label_val": label_out,
                "valid":     date_valid & label_valid & plugin_ok,
                "p_idx":     p_idx_map,
            }

        # EFFICIENCY: instead of building each day's batch from Python lists with
        # D × T × P iterations + torch.stack, we pre-build a (D, T, *shape)
        # tensor grid for each modality once, then the assembly loop is a single
        # boolean-index slice per day — no Python loops over tickers at all.
        logger.info("📦 Building tensor grid for fast assembly...")

        D = len(dates)
        T = len(universe)
        ticker_to_idx = {t: i for i, t in enumerate(universe)}

        # Determine feature shapes from the first valid ticker
        ref_shapes: dict = {}
        for t in universe:
            tf = precomputed_features.get(t)
            if not tf:
                continue
            for p_name in plugin_names:
                if p_name in tf and p_name not in ref_shapes:
                    ref_shapes[p_name] = tf[p_name][1].shape[1:]  # (*shape)
            if len(ref_shapes) == len(plugin_names):
                break

        # Allocate grids
        valid_grid = np.zeros((D, T), dtype=bool)
        label_grid = np.full((D, T), np.nan, dtype=np.float32)
        price_grid = np.zeros((D, T), dtype=np.float32)
        feat_grids: dict = {}
        for p_name in plugin_names:
            if p_name not in ref_shapes:
                continue
            f_key = p_key_map[p_name]
            feat_grids[f_key] = torch.zeros(D, T, *ref_shapes[p_name])

        # Fill grids per ticker (each ticker touches only its valid rows)
        for ticker in universe:
            ti = ticker_info.get(ticker)
            if ti is None:
                continue
            t_i = ticker_to_idx[ticker]
            valid_days = np.where(ti["valid"])[0]
            if len(valid_days) == 0:
                continue

            valid_grid[valid_days, t_i] = True
            label_grid[valid_days, t_i] = ti["label_val"][valid_days]

            # Prices
            if ticker in precomputed_prices:
                pr_ns, pr_arr = precomputed_prices[ticker]
                actual_ns_v    = ti["actual_ns"][valid_days]
                pr_idxs        = np.minimum(
                    np.searchsorted(pr_ns, actual_ns_v, side="left"), len(pr_ns) - 1
                )
                match = pr_ns[pr_idxs] == actual_ns_v
                price_grid[valid_days[match], t_i] = pr_arr[pr_idxs[match]]

            # Features — one advanced-index assign per (ticker, modality)
            for p_name in plugin_names:
                f_key = p_key_map.get(p_name)
                if f_key not in feat_grids:
                    continue
                if p_name not in precomputed_features.get(ticker, {}):
                    continue
                _, p_tensors = precomputed_features[ticker][p_name]
                p_idxs = ti["p_idx"][p_name][valid_days]
                vd_t   = torch.from_numpy(valid_days.astype(np.int64))
                feat_grids[f_key][vd_t, t_i] = p_tensors[p_idxs]

        del precomputed_features, precomputed_prices
        gc.collect()

        # Times grid (only needed for valid entries)
        times_grid = np.zeros((D, T), dtype=np.int64)
        for ticker in universe:
            ti = ticker_info.get(ticker)
            if ti is None:
                continue
            t_i = ticker_to_idx[ticker]
            valid_days = np.where(ti["valid"])[0]
            times_grid[valid_days, t_i] = ti["actual_ns"][valid_days]

        logger.info("📦 Assembling Multi-Modal Batches...")
        for d_i, d in enumerate(tqdm(dates, desc="🏗️ Building Sniper Batches")):
            valid_ts = np.where(valid_grid[d_i])[0]
            if len(valid_ts) == 0:
                continue

            batch_tickers = [universe[t_i] for t_i in valid_ts]
            vt_tensor     = torch.from_numpy(valid_ts.astype(np.int64))

            batch_data: dict = {}
            # price: (T_valid,) — direct index into pre-built grid
            batch_data["raw_price"] = torch.from_numpy(price_grid[d_i, valid_ts])
            # features: (T_valid, *shape) — one slice per modality, no ticker loop
            for f_key, grid in feat_grids.items():
                batch_data[f_key] = grid[d_i, vt_tensor]

            batch = MultiModalBatch(
                data=batch_data,
                labels=torch.from_numpy(label_grid[d_i, valid_ts]),
                tickers=batch_tickers,
                times=[pd.Timestamp(int(times_grid[d_i, t_i])) for t_i in valid_ts],
            )
            results.append({"date": d, "batch": batch})
                
        logger.success(f"✅ Fast Sniper Dataset Built: {len(results)} days.")
        
        try:
            os.makedirs('data', exist_ok=True)
            torch.save(results, cache_path)
            logger.info(f"💾 Walk-Forward cache saved to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save Walk-Forward cache: {e}")
            
        return results