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
from research_lab.plugins.core_plugins import SequentialPlugin, SpatialPlugin, GraphPlugin
from research_lab.real_data_ingestor import InstitutionalIngestor
from research_lab.data_engine import DataEngine
from scipy.stats import spearmanr

class BacktestOrchestrator:
    def __init__(self, tickers=None):
        with open("config.yaml", "r") as f: self.config = yaml.safe_load(f)
        db_path = self.config['data_engine']['db_path']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = DataEngine(storage_path=db_path)
        self.tickers = tickers if tickers else self.config['universe']['tickers']
        try:
            count = self.engine.conn.execute("SELECT count(*) FROM market_data").fetchone()[0]
            if count == 0: logger.error(f"❌ DATABASE IS EMPTY ({db_path}). Please run with --ingest first.")
            else: logger.info(f"✅ Pre-flight Check: Found {count} records in DuckDB.")
        except Exception as e: logger.warning(f"⚠️ Pre-flight database check failed: {e}.")
        d_param = self.config['signal_physics']['fractional_differentiation']['d_param']
        scales = np.array(self.config['signal_physics']['wavelet_transform']['scales'])
        self.lookback, self.horizon = self.config['signal_physics']['lookback_days'], self.config['signal_physics']['horizon_days']
        self.plugins = [SequentialPlugin(d_param=d_param), SpatialPlugin(scales=scales), GraphPlugin(feature_dim=8)]
        self.universe = AlphaUniverse(data_provider=self.engine, plugins=self.plugins, config=self.config)
        self.n_scales = len(scales)

    def run_ingestion(self):
        logger.info("📡 INITIATING DATA INGESTION...")
        ingestor = InstitutionalIngestor(self.engine)
        ingestor.ingest_universe(self.tickers, self.config['data_engine']['start_date'], datetime.now().strftime("%Y-%m-%d"))

    def _collect_training_data(self, start: datetime, end: datetime) -> MultiModalBatch:
        logger.info(f"📥 Collecting multi-regime training data ({start.date()} -> {end.date()})...")
        # REVERT: Use latest_only=False to restore the 1.8M sample count
        steps = self.universe.walk_forward(universe=self.tickers, start_date=start, end_date=end, stride=1, lookback=self.lookback, horizon=self.horizon, latest_only=False)
        if not steps: logger.error("❌ NO TRAINING SAMPLES PRODUCED."); raise ValueError("Empty dataset.")
        all_x_seq, all_x_spatial, all_x_graph, all_y = [], [], [], []
        for step in steps:
            batch = step['batch']
            all_x_seq.append(batch.data['x_seq']); all_x_spatial.append(batch.data['x_spatial']); all_x_graph.append(batch.data['x_graph']); all_y.append(batch.labels)
        return MultiModalBatch(
            data={'x_seq': torch.cat(all_x_seq), 'x_spatial': torch.cat(all_x_spatial), 'x_graph': torch.cat(all_x_graph)},
            labels=torch.cat(all_y), tickers=['COMBINED'] * len(torch.cat(all_y)), times=[datetime.now()] * len(torch.cat(all_y))
        )

    def run_comparison(self, train_start: datetime, train_end: datetime, test_start: datetime, test_end: datetime, skip_train: bool = False):
        logger.info("📊 INITIATING PRODUCTION TRIAL...")
        arch_config = self.config['model_pipeline']['architecture']
        hidden_dim, vh, gl = arch_config['hidden_dim'], arch_config['vit_heads'], arch_config['gnn_layers']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = self.config['model_pipeline']['training'].get('batch_size', 8192)
        wd = self.config['model_pipeline']['training'].get('weight_decay', 0.0001)
        do = self.config['model_pipeline']['training'].get('dropout', 0.3)
        champion = RankNet(input_dim=self.n_scales, hidden_dim=hidden_dim)
        specs = [InputSpec(name='x_seq', shape=(self.lookback, 1), type='seq'), InputSpec(name='x_spatial', shape=(1, self.n_scales, self.lookback), type='spatial'), InputSpec(name='x_graph', shape=(8,), type='graph')]
        challenger = MultiModalRankNet(specs=specs, hidden_dim=hidden_dim, vit_heads=vh, gnn_layers=gl, dropout=do)
        using_subset = len(self.tickers) < len(self.config['universe']['tickers'])
        if not skip_train:
            cache_path = "data/train_dataset.pt"
            if os.path.exists(cache_path) and not using_subset:
                logger.info(f"📦 LOADING CACHED TRAINING FEATURES: {cache_path}")
                torch.serialization.add_safe_globals([MultiModalBatch, datetime]); train_dataset = torch.load(cache_path, weights_only=True)
                if train_dataset.data['x_spatial'].shape[2] != self.n_scales: logger.warning("⚠️ Cache mismatch. RE-COLLECTING..."); train_dataset = self._collect_training_data(train_start, train_end); torch.save(train_dataset, cache_path)
            else: 
                train_dataset = self._collect_training_data(train_start, train_end)
                if not using_subset: torch.save(train_dataset, cache_path)
            nt = len(train_dataset.labels); n_train = int(nt * 0.8); idx = torch.randperm(nt)
            val_dataset = MultiModalBatch(data={k: v[idx[n_train:]] for k, v in train_dataset.data.items()}, labels=train_dataset.labels[idx[n_train:]], tickers=['VAL'] * (nt - n_train), times=[datetime.now()] * (nt - n_train))
            train_dataset = MultiModalBatch(data={k: v[idx[:n_train]] for k, v in train_dataset.data.items()}, labels=train_dataset.labels[idx[:n_train]], tickers=['TRAIN'] * n_train, times=[datetime.now()] * n_train)
            if device.type == 'cuda': logger.info("🚀 MOVING DATASETS TO GPU VRAM..."); train_dataset = train_dataset.to(device); val_dataset = val_dataset.to(device)
            logger.info(f"🛠️ Training on {len(train_dataset.labels)} samples with {len(val_dataset.labels)} validation hold-out...")
            ep, lr, pat = self.config['model_pipeline']['training']['epochs'], self.config['model_pipeline']['training']['lr'], self.config['model_pipeline']['training'].get('early_stopping_patience', 5)
            logger.info("🏆 Training Champion (MLP)...")
            champion.fit((train_dataset.data['x_spatial'][:, 0, :, -1], train_dataset.labels), epochs=ep, lr=lr, device=device, batch_size=batch_size, patience=pat, weight_decay=wd)
            logger.info("🚀 Training Challenger (Multi-Modal)...")
            challenger.fit(train_dataset, epochs=ep, lr=lr, device=device, batch_size=batch_size, patience=pat, weight_decay=wd, val_dataset=val_dataset)
            os.makedirs("models", exist_ok=True); challenger.export("models/challenger_v1.pt"); torch.save(champion.state_dict(), "models/champion_baseline.pt")
            logger.success("✅ Training complete. Models exported.")
        else:
            logger.info("⏭️ SKIPPING TRAINING. Loading models..."); torch.serialization.add_safe_globals([MultiModalBatch, datetime]); challenger = torch.jit.load("models/challenger_v1.pt").to(device)
            if os.path.exists("models/champion_baseline.pt"): champion.load_state_dict(torch.load("models/champion_baseline.pt", weights_only=True))
            champion.to(device)
        logger.info(f"🧪 Evaluating on OOS regime ({test_start.date()} -> {test_end.date()})...")
        oos_cache_path = "data/oos_steps.pt"
        if os.path.exists(oos_cache_path) and not using_subset: logger.info(f"📦 LOADING CACHED BACKTEST DATA: {oos_cache_path}"); steps = torch.load(oos_cache_path, weights_only=True)
        else: steps = self.universe.walk_forward(universe=self.tickers, start_date=test_start, end_date=test_end, stride=21, lookback=self.lookback, horizon=5, latest_only=True)
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
