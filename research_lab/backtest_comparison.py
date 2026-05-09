import torch
import numpy as np
import pandas as pd
import yaml
import os
import argparse
from datetime import datetime, timedelta
from qts_core.logger import logger
from research_lab.alpha_universe import AlphaUniverse, MultiModalBatch
from research_lab.alpha_ranker import RankNet, MultiModalRankNet, InputSpec
from research_lab.real_data_ingestor import InstitutionalIngestor
from research_lab.data_engine import DataEngine
from scipy.stats import spearmanr

class BacktestOrchestrator:
    def __init__(self, tickers=None):
        with open("config.yaml", "r") as f: self.config = yaml.safe_load(f)
        db_path = self.config['data_engine']['storage_path']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = DataEngine(storage_path=db_path)
        self.tickers = tickers if tickers else self.config['universe']['tickers']
        
        scales = np.array(self.config['signal_physics']['wavelet_transform']['scales'])
        self.lookback, self.horizon = self.config['signal_physics']['lookback_days'], self.config['signal_physics']['horizon_days']
        # SENIOR FIX: Pass the shared connection to enable safe parallelization
        self.universe = AlphaUniverse(conn=self.engine.conn, config=self.config)
        self.n_scales = len(scales)

    def run_ingestion(self):
        logger.info("📡 INITIATING DATA INGESTION...")
        # SENIOR DEV FIX: Explicitly pass env variables to ensure they aren't lost in sub-processes
        env_keys = {
            'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
            'ALPACA_API_SECRET': os.getenv('ALPACA_API_SECRET'),
            'TIINGO_API_KEY': os.getenv('TIINGO_API_KEY'),
            'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY')
        }
        ingestor = InstitutionalIngestor(self.engine, config=self.config)
        # Inject keys directly into the ingestor instance
        for k, v in env_keys.items(): 
            if v: setattr(ingestor, k.lower() if 'API' not in k else k.lower(), v)
            
        ingestor.ingest_universe(self.tickers, self.config['data_engine']['start_date'], datetime.now().strftime("%Y-%m-%d"))

    def _collect_training_data(self, start: datetime, end: datetime) -> MultiModalBatch:
        logger.info(f"📥 Collecting multi-regime training data ({start.date()} -> {end.date()})...")
        
        # SENIOR FIX: Explicitly close the main connection to release the DuckDB lock.
        # This is mandatory before Parallel walk_forward starts.
        self.engine.close()
        
        # SENIOR FIX: Use walk_forward to avoid OOM/Hang on large intraday datasets.
        # stride=1 day ensures we get one sample per trading day.
        # latest_only=True + the new 16:00 alignment in snapshot() ensures high fidelity.
        steps = self.universe.walk_forward(universe=self.tickers, start_date=start, end_date=end, stride=1, latest_only=True, backtest_mode=True)
        if not steps: logger.error("❌ NO TRAINING SAMPLES PRODUCED."); raise ValueError("Empty dataset.")
        
        modalities = list(steps[0]['batch'].data.keys())
        all_data, all_y, all_times = {m: [] for m in modalities}, [], []
        for step in steps:
            batch = step['batch']
            for m in modalities: all_data[m].append(batch.data[m])
            all_y.append(batch.labels); all_times.extend(batch.times)
            
        return MultiModalBatch(
            data={m: torch.cat(all_data[m]) for m in modalities},
            labels=torch.cat(all_y), tickers=['COMBINED'] * len(torch.cat(all_y)), times=all_times
        )

    def run_comparison(self, train_start: datetime, train_end: datetime, test_start: datetime, test_end: datetime, skip_train: bool = False):
        logger.info("📊 INITIATING PRODUCTION TRIAL...")
        arch_config = self.config['model_pipeline']['architecture']
        hidden_dim, vh, gl = arch_config['hidden_dim'], arch_config['vit_heads'], arch_config['gnn_layers']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = self.config['model_pipeline']['training'].get('batch_size', 1024)
        wd = self.config['model_pipeline']['training'].get('weight_decay', 0.0002)
        do = self.config['model_pipeline']['training'].get('dropout', 0.5)
        model_path = self.config['model_pipeline'].get('model_path', 'models/challenger_v2.pt')
        
        specs = []
        for p in self.universe.plugins:
            if p.name == 'x_seq': specs.append(InputSpec(name='x_seq', shape=(self.lookback, 1), type='seq'))
            elif p.name == 'x_spatial': specs.append(InputSpec(name='x_spatial', shape=(1, self.n_scales, self.lookback), type='spatial'))
            elif p.name == 'x_graph': specs.append(InputSpec(name='x_graph', shape=(8,), type='graph'))
            elif p.name == 'x_volume': specs.append(InputSpec(name='x_volume', shape=(self.lookback, 1), type='seq'))

        champion = RankNet(input_dim=self.n_scales, hidden_dim=hidden_dim)
        challenger = MultiModalRankNet(specs=specs, hidden_dim=hidden_dim, vit_heads=vh, gnn_layers=gl, dropout=do)
        
        using_subset = len(self.tickers) < 50
        from datetime import datetime as dt_class
        torch.serialization.add_safe_globals([MultiModalBatch, dt_class, pd.Timestamp, pd._libs.tslibs.timestamps._unpickle_timestamp])

        if not skip_train:
            cache_path = self.config['model_pipeline'].get('training_dataset_path', 'data/train_dataset_v2_full.pt')
            if os.path.exists(cache_path) and not using_subset:
                logger.info(f"📦 LOADING CACHED TRAINING FEATURES: {cache_path}")
                train_dataset = torch.load(cache_path, weights_only=True)
                # SENIOR FIX: Validate cache compatibility (modalities AND scale count)
                cache_scales = train_dataset.data['x_spatial'].shape[2] if 'x_spatial' in train_dataset.data else 0
                if len(train_dataset.data) != len(specs) or cache_scales != self.n_scales:
                     logger.warning("⚠️ Cache mismatch detected (scales or modalities changed). Re-generating training data...")
                     train_dataset = self._collect_training_data(train_start, train_end); torch.save(train_dataset, cache_path)
            else: 
                train_dataset = self._collect_training_data(train_start, train_end)
                if not using_subset: torch.save(train_dataset, cache_path)
            
            # AUTOMATIC VALIDATION SELECTION (Relative to train_end)
            val_pct = self.config['model_pipeline']['timeframes'].get('validation_split_pct', 0.15)
            # Find the date that marks the start of the last val_pct of samples
            sorted_times = sorted(train_dataset.times)
            cutoff_idx = int(len(sorted_times) * (1 - val_pct))
            cutoff_date = sorted_times[cutoff_idx]
            
            logger.info(f"📊 Auto-Validation Cutoff: {cutoff_date.date()} (Last {val_pct*100:.0f}%)")
            
            train_mask = [t < cutoff_date for t in train_dataset.times]
            val_mask = [t >= cutoff_date for t in train_dataset.times]
            val_dataset = MultiModalBatch(data={k: v[val_mask] for k, v in train_dataset.data.items()}, labels=train_dataset.labels[val_mask], tickers=['VAL'] * sum(val_mask), times=[t for i, t in enumerate(train_dataset.times) if val_mask[i]])
            train_dataset = MultiModalBatch(data={k: v[train_mask] for k, v in train_dataset.data.items()}, labels=train_dataset.labels[train_mask], tickers=['TRAIN'] * sum(train_mask), times=[t for i, t in enumerate(train_dataset.times) if train_mask[i]])
            
            if device.type == 'cuda': logger.info("🚀 MOVING DATASETS TO GPU VRAM..."); train_dataset = train_dataset.to(device); val_dataset = val_dataset.to(device)
            logger.info(f"🛠️ Training on {len(train_dataset.labels)} samples with {len(val_dataset.labels)} TEMPORAL validation (Late 2022)...")
            ep, lr, pat = self.config['model_pipeline']['training']['epochs'], self.config['model_pipeline']['training']['lr'], self.config['model_pipeline']['training'].get('early_stopping_patience', 5)
            logger.info("🏆 Training Champion (MLP)...")
            champion.fit((train_dataset.data['x_spatial'][:, 0, :, -1], train_dataset.labels), epochs=ep, lr=lr, device=device, batch_size=batch_size, patience=pat, weight_decay=wd)
            logger.info("🚀 Training Challenger (Multi-Modal)...")
            challenger.fit(train_dataset, epochs=ep, lr=lr, device=device, batch_size=batch_size, patience=pat, weight_decay=wd, val_dataset=val_dataset)
            os.makedirs("models", exist_ok=True); challenger.export(model_path); torch.save(champion.state_dict(), "models/champion_baseline_v2.pt")
            logger.success(f"✅ Training complete. Models exported to {model_path}")
        else:
            logger.info(f"⏭️ SKIPPING TRAINING. Loading models from {model_path}...")
            challenger = torch.jit.load(model_path).to(device)
            if os.path.exists("models/champion_baseline_v2.pt"): champion.load_state_dict(torch.load("models/champion_baseline_v2.pt", weights_only=True))
            champion.to(device)

        logger.info(f"🧪 Evaluating on OOS regime ({test_start.date()} -> {test_end.date()})...")
        oos_cache_path = self.config['model_pipeline'].get('evaluation_dataset_path', 'data/oos_steps.pt')
        if os.path.exists(oos_cache_path) and not using_subset: 
            logger.info(f"📦 LOADING CACHED BACKTEST DATA: {oos_cache_path}")
            steps = torch.load(oos_cache_path, weights_only=True)
            # SENIOR FIX: Validate OOS cache compatibility (modalities AND scale count)
            if steps:
                batch = steps[0]['batch']
                cache_scales = batch.data['x_spatial'].shape[2] if 'x_spatial' in batch.data else 0
                cache_modalities = list(batch.data.keys())
                spec_modalities = [s.name for s in specs]
                
                mismatch = False
                if cache_scales != self.n_scales:
                    logger.warning(f"⚠️ OOS Cache scale mismatch: Cache has {cache_scales}, Config has {self.n_scales}")
                    mismatch = True
                elif set(cache_modalities) != set(spec_modalities):
                    logger.warning(f"⚠️ OOS Cache modality mismatch: Cache {cache_modalities}, Specs {spec_modalities}")
                    mismatch = True
                    
                if mismatch:
                    logger.info("🔄 Re-generating OOS evaluation data...")
                    self.engine.close()
                    steps = self.universe.walk_forward(universe=self.tickers, start_date=test_start, end_date=test_end, stride=21, lookback=self.lookback, horizon=5, latest_only=True)
                    if not using_subset: torch.save(steps, oos_cache_path)
        else: 
            # SENIOR FIX: Release lock
            self.engine.close()
            steps = self.universe.walk_forward(universe=self.tickers, start_date=test_start, end_date=test_end, stride=21, lookback=self.lookback, horizon=5, latest_only=True)
            if not using_subset: torch.save(steps, oos_cache_path)
            
        res = []
        for step in steps:
            batch = step['batch'].to(device) if device.type == 'cuda' else step['batch']
            with torch.no_grad():
                champion.to(device); c_scores = champion(batch.data['x_spatial'][:, 0, :, -1]).squeeze().cpu().numpy()
                if hasattr(challenger, 'predict_dataset'): ch_scores = challenger.predict_dataset(batch, batch_size=batch_size).squeeze().cpu().numpy()
                else: ch_scores = challenger({k: v.to(device) for k, v in batch.data.items()}).squeeze().cpu().numpy()
            target = batch.labels.cpu().numpy(); c_ic = 0 if np.std(c_scores) < 1e-6 or np.std(target) < 1e-6 else spearmanr(c_scores, target)[0]; ch_ic = 0 if np.std(ch_scores) < 1e-6 or np.std(target) < 1e-6 else spearmanr(ch_scores, target)[0]
            res.append({'date': step['date'], 'champion_ic': c_ic if not np.isnan(c_ic) else 0, 'challenger_ic': ch_ic if not np.isnan(ch_ic) else 0})
        df = pd.DataFrame(res); df.to_csv("data/backtest_results.csv", index=False)
        logger.info(f"\n--- Backtest Summary ---\nChampion Avg IC: {df['champion_ic'].mean():.4f}\nChallenger Avg IC: {df['challenger_ic'].mean():.4f}\nWin Rate: {(df['challenger_ic'] > df['champion_ic']).mean()*100:.1f}%")
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UQTS-2026 Production Backtest Orchestrator")
    parser.add_argument("--ingest", action="store_true"); parser.add_argument("--train", action="store_true"); parser.add_argument("--eval-only", action="store_true"); parser.add_argument("--test-subset", action="store_true")
    args = parser.parse_args(); orchestrator = BacktestOrchestrator(tickers=["SPY", "NVDA", "TSM"] if args.test_subset else None)
    if args.ingest: orchestrator.run_ingestion()
    if args.train or (not args.ingest and not args.eval_only): orchestrator.run_comparison(datetime(2018, 1, 1), datetime(2022, 12, 31), datetime(2023, 1, 1), datetime.now(), skip_train=args.eval_only)
