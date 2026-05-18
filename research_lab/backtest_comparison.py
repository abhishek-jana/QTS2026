import torch
import numpy as np
import pandas as pd
import yaml
import os
import argparse
from datetime import datetime, timedelta
from qts_core.logger import logger
from research_lab.alpha_universe import AlphaUniverse, MultiModalBatch
from research_lab.alpha_ranker_sniper import SniperRanker
from research_lab.alpha_ranker import RankNet # Keep for baseline
from research_lab.real_data_ingestor import InstitutionalIngestor
from research_lab.data_engine import DataEngine
from scipy.stats import spearmanr
from tqdm import tqdm

class BacktestOrchestrator:
    def __init__(self, tickers=None):
        with open("config.yaml", "r") as f: self.config = yaml.safe_load(f)
        db_path = self.config['data_engine']['storage_path']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = DataEngine(storage_path=db_path)
        self.tickers = tickers if tickers else self.config['universe']['tickers']
        
        scales = np.array(self.config['signal_physics']['wavelet_transform']['scales'])
        self.lookback, self.horizon = self.config['signal_physics']['lookback_days'], self.config['signal_physics']['horizon_days']
        self.universe = AlphaUniverse(conn=self.engine.conn, config=self.config)
        self.n_scales = len(scales)

    def run_ingestion(self):
        logger.info("📡 INITIATING DATA INGESTION...")
        env_keys = {
            'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
            'ALPACA_API_SECRET': os.getenv('ALPACA_API_SECRET'),
            'TIINGO_API_KEY': os.getenv('TIINGO_API_KEY'),
            'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY')
        }
        ingestor = InstitutionalIngestor(self.engine, config=self.config)
        for k, v in env_keys.items(): 
            if v: setattr(ingestor, k.lower() if 'API' not in k else k.lower(), v)
            
        ingestor.ingest_universe(self.tickers, self.config['data_engine']['start_date'], "now")

    def _collect_training_data(self, start: datetime, end: datetime) -> MultiModalBatch:
        logger.info(f"📥 Collecting Sniper training data ({start.date()} -> {end.date()})...")
        steps = self.universe.walk_forward(universe=self.tickers, start_date=start, end_date=end, stride=1)
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
        logger.info("📊 INITIATING SNIPER PRODUCTION TRIAL...")
        arch_config = self.config['model_pipeline']['architecture']
        hidden_dim = arch_config.get('hidden_dim', 32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = self.config['model_pipeline']['training'].get('batch_size', 128)
        model_path = self.config['model_pipeline'].get('model_path', 'models/sniper_v7.pt')
        
        # Define TFT Specs based on AlphaUniverse snap keys
        # AlphaUniverse uses x_static_ and x_past_ prefixes
        specs = {
            'static': {'x_static': 1}, # From StaticMetadataPlugin
            'past': {
                'x_seq': 1,
                'x_spatial': self.n_scales, # Wavelet scales
                'x_volume': 1,
                'x_momentum': 3,
                'x_calendar': 4
            }
        }

        champion = RankNet(input_dim=self.n_scales, hidden_dim=64) # Baseline
        challenger = SniperRanker(specs=specs, hidden_dim=hidden_dim)
        
        using_subset = len(self.tickers) < 10
        from datetime import datetime as dt_class
        torch.serialization.add_safe_globals([MultiModalBatch, dt_class, pd.Timestamp, pd._libs.tslibs.timestamps._unpickle_timestamp])

        if not skip_train:
            cache_path = self.config['model_pipeline'].get('training_dataset_path', 'data/train_dataset_v7.pt')
            if os.path.exists(cache_path) and not using_subset:
                logger.info(f"📦 LOADING CACHED SNIPER FEATURES: {cache_path}")
                train_dataset = torch.load(cache_path, weights_only=True)
            else: 
                train_dataset = self._collect_training_data(train_start, train_end)
                if not using_subset: torch.save(train_dataset, cache_path)
            
            # Temporal Split
            val_pct = self.config['model_pipeline']['timeframes'].get('validation_split_pct', 0.15)
            sorted_times = sorted(train_dataset.times)
            cutoff_date = sorted_times[int(len(sorted_times) * (1 - val_pct))]
            
            train_mask = [t < cutoff_date for t in train_dataset.times]
            val_mask = [t >= cutoff_date for t in train_dataset.times]
            
            val_dataset = MultiModalBatch(
                data={k: v[val_mask] for k, v in train_dataset.data.items()}, 
                labels=train_dataset.labels[val_mask], tickers=[], times=[]
            )
            train_dataset = MultiModalBatch(
                data={k: v[train_mask] for k, v in train_dataset.data.items()}, 
                labels=train_dataset.labels[train_mask], tickers=[], times=[]
            )

            logger.info(f"🚀 Training Sniper (TFT) on {len(train_dataset.labels)} samples...")
            patience = self.config['model_pipeline']['training'].get('early_stopping_patience', 15)
            challenger.fit(train_dataset, epochs=self.config['model_pipeline']['training']['epochs'], 
                           lr=self.config['model_pipeline']['training']['lr'], 
                           device=device, batch_size=batch_size, val_dataset=val_dataset,
                           patience=patience)
            
            os.makedirs("models", exist_ok=True); torch.save(challenger.state_dict(), model_path)
            logger.success(f"✅ Sniper trained and saved to {model_path}")
        else:
            logger.info(f"⏭️ LOADING SNIPER FROM {model_path}...")
            challenger.load_state_dict(torch.load(model_path, map_location=device))
            challenger.to(device)

        # OOS Evaluation
        logger.info(f"🧪 Evaluating on OOS regime ({test_start.date()} -> {test_end.date()})...")
        steps = self.universe.walk_forward(universe=self.tickers, start_date=test_start, end_date=test_end, stride=1)
            
        res = []
        for step in tqdm(steps, desc="🔍 Sniper In-Action (IC)"):
            batch = step['batch'].to(device)
            with torch.no_grad():
                ch_scores = challenger.predict(batch)[:, 1].cpu().numpy() # Median Quantile
            
            target = batch.labels.cpu().numpy()
            ic = spearmanr(ch_scores, target)[0] if np.std(ch_scores) > 1e-9 else 0.0
            res.append({'date': step['date'], 'sniper_ic': ic})
        
        df = pd.DataFrame(res)
        logger.info(f"🏁 SNIPER OOS AVG IC: {df['sniper_ic'].mean():.4f}")
        
        # --- FINANCIAL BACKTEST ---
        logger.info("💸 Running Financial Backtest Simulation (Optimized In-Memory)...")
        from execution_muscle.inference_worker import InferenceWorker
        
        worker = InferenceWorker()
        worker.trading_mode = 'sim'  # Force sim mode for backtest
        worker.rl_pilot = None       # SENIOR FIX: Ensure baseline is signal-only
        worker.initialize()
        
        # Sync the worker to use the model we just trained/loaded
        worker.strategy.model = challenger
        worker.current_knowledge_time = test_start.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # PRE-LOAD ALL DATA INTO MEMORY FOR BLAZING FAST SIMULATION
        logger.info("Pre-loading historical prices for OMS and Risk Parity...")
        fetch_start = test_start - timedelta(days=100)
        all_history = worker.strategy.lab.get_batch_pit_view(worker.tickers, as_of=test_end + timedelta(days=5), start_time=fetch_start)
        all_history['knowledge_time'] = pd.to_datetime(all_history['knowledge_time'])
        all_history.sort_values('knowledge_time', inplace=True)
        history_by_ticker = {t: df for t, df in all_history.groupby('ticker')}
        
        def fast_get_pit_view(ticker, as_of):
            if ticker not in history_by_ticker: return pd.DataFrame()
            df = history_by_ticker[ticker]
            return df[df['knowledge_time'] <= as_of]
            
        def fast_get_latest_price(ticker):
            view = fast_get_pit_view(ticker, worker.current_knowledge_time)
            if not view.empty: return float(view['close'].iloc[-1])
            return 0.0

        # Inject fast paths
        worker.data_engine.get_pit_view = fast_get_pit_view
        worker._get_latest_price_sim = fast_get_latest_price
        
        days_processed = 0
        total_days = len(steps)
        
        for step in steps:
            worker.current_knowledge_time = step['date']
            
            # Temporarily inject the batch_override logic via a wrapper
            original_snapshot = worker.strategy.lab.snapshot
            worker.strategy.lab.snapshot = lambda **kwargs: step['batch']
            
            house_view = worker.strategy.get_current_rankings(as_of=worker.current_knowledge_time, include_batch=False)
            
            # Restore snapshot method
            worker.strategy.lab.snapshot = original_snapshot
            
            if house_view['status'] == 'OK':
                stats = worker._update_oms_sim(house_view)
                if stats is None:
                    logger.warning(f"Sim step @ {worker.current_knowledge_time} failed to return stats (Mode: {worker.trading_mode}). Using previous NLV.")
                    last_nlv = worker.performance_history[-1]['portfolio'] if worker.performance_history else 100000.0
                    stats = {'nlv': last_nlv}

                spy_p = worker._get_latest_price_sim('SPY')
                if not hasattr(worker, 'spy_start_p'): worker.spy_start_p = spy_p
                spy_cap = (spy_p / worker.spy_start_p) * 100000.0 if worker.spy_start_p > 0 else 100000.0
                
                worker.performance_history.append({
                    'time': worker.current_knowledge_time.strftime('%Y-%m-%d'), 
                    'portfolio': float(stats['nlv']), 
                    'spy': float(spy_cap)
                })
            
            days_processed += 1
            if days_processed % 30 == 0 and len(worker.performance_history) > 0:
                logger.info(f"Simulated up to {worker.current_knowledge_time.strftime('%Y-%m-%d')} | Sniper NLV: ${worker.performance_history[-1]['portfolio']:,.2f} | SPY NLV: ${worker.performance_history[-1]['spy']:,.2f}")
            
        if worker.performance_history:
            final_nlv = worker.performance_history[-1]['portfolio']
            spy_nlv = worker.performance_history[-1]['spy']
            
            sniper_ret = (final_nlv / 100000.0) - 1.0
            spy_ret = (spy_nlv / 100000.0) - 1.0
            
            logger.success("================================================")
            logger.success(f"📈 BACKTEST COMPLETE: {test_start.date()} to {test_end.date()}")
            logger.success(f"🎯 Sniper V7.0 Return: {sniper_ret*100:.2f}% (${final_nlv:,.2f})")
            logger.success(f"📉 S&P 500 Return:     {spy_ret*100:.2f}% (${spy_nlv:,.2f})")
            logger.success(f"🏆 Alpha Generated:    {(sniper_ret - spy_ret)*100:.2f}%")
            logger.success("================================================")
        
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sniper Backtest Orchestrator")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--test-subset", action="store_true")
    args = parser.parse_args()
    
    orchestrator = BacktestOrchestrator(tickers=["SPY", "NVDA", "AAPL"] if args.test_subset else None)
    if args.ingest: orchestrator.run_ingestion()
    
    tf = orchestrator.config['model_pipeline']['timeframes']
    def p(s): return datetime.now() if s == 'now' else datetime.strptime(s, '%Y-%m-%d')
    
    horizon = orchestrator.config['signal_physics'].get('horizon_days', 3)
    clean_train_end = p(tf['train_end']) - timedelta(days=horizon)
    
    orchestrator.run_comparison(p(tf['train_start']), clean_train_end, p(tf['test_start']), p(tf['test_end']), skip_train=args.eval_only)
