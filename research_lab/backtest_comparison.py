import torch
import numpy as np
import pandas as pd
import yaml
import os
import argparse
from datetime import datetime, timedelta
from research_lab.alpha_universe import AlphaUniverse, MultiModalBatch
from research_lab.alpha_ranker import RankNet, MultiModalRankNet, InputSpec
from research_lab.plugins.core_plugins import SequentialPlugin, SpatialPlugin, GraphPlugin
from research_lab.real_data_ingestor import InstitutionalIngestor
from research_lab.data_engine import DataEngine
from scipy.stats import spearmanr

class BacktestOrchestrator:
    """
    Executes and compares 'Champion' vs 'Challenger' models on real data.
    """
    def __init__(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        db_path = self.config['data_engine']['db_path']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = DataEngine(storage_path=db_path)
        
        self.tickers = self.config['universe']['tickers']
        
        # 1. Properly pull Signal Physics parameters
        d_param = self.config['signal_physics']['fractional_differentiation']['d_param']
        scales = np.array(self.config['signal_physics']['wavelet_transform']['scales'])
        self.lookback = self.config['signal_physics']['lookback_days']
        self.horizon = self.config['signal_physics']['horizon_days']

        self.plugins = [
            SequentialPlugin(d_param=d_param), 
            SpatialPlugin(scales=scales),
            GraphPlugin(feature_dim=8)
        ]
        self.universe = AlphaUniverse(data_provider=self.engine, plugins=self.plugins, config=self.config)
        self.n_scales = len(scales)

    def run_ingestion(self):
        """Standalone function for data download/caching."""
        print(f"📡 INITIATING DATA INGESTION...")
        ingestor = InstitutionalIngestor(self.engine)
        today_str = datetime.now().strftime("%Y-%m-%d")
        ingestor.ingest_universe(self.tickers, self.config['data_engine']['start_date'], today_str)

    def _collect_training_data(self, start: datetime, end: datetime) -> MultiModalBatch:
        """Walks forward through the training regime to build a large dataset."""
        print(f"📥 Collecting multi-regime training data ({start.date()} -> {end.date()})...")
        steps = self.universe.walk_forward(
            universe=self.tickers,
            start_date=start,
            end_date=end,
            stride=21, # Monthly snapshots for training speed
            lookback=self.lookback,
            horizon=self.horizon
        )
        
        all_x_seq, all_x_spatial, all_x_graph, all_y = [], [], [], []
        for step in steps:
            batch = step['batch']
            all_x_seq.append(batch.data['x_seq'])
            all_x_spatial.append(batch.data['x_spatial'])
            all_x_graph.append(batch.data['x_graph'])
            all_y.append(batch.labels)
            
        return MultiModalBatch(
            data={'x_seq': torch.cat(all_x_seq), 'x_spatial': torch.cat(all_x_spatial), 'x_graph': torch.cat(all_x_graph)},
            labels=torch.cat(all_y),
            tickers=['COMBINED'] * sum(len(b.labels) for b in [s['batch'] for s in steps]),
            times=[datetime.now()] * sum(len(b.labels) for b in [s['batch'] for s in steps])
        )

    def run_comparison(self, train_start: datetime, train_end: datetime, test_start: datetime, test_end: datetime, skip_train: bool = False):
        print(f"📊 INITIATING PRODUCTION TRIAL...")
        arch_config = self.config['model_pipeline']['architecture']
        hidden_dim, vit_heads, gnn_layers = arch_config['hidden_dim'], arch_config['vit_heads'], arch_config['gnn_layers']
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = self.config['model_pipeline']['training'].get('batch_size', 8192)
        
        champion = RankNet(input_dim=self.n_scales, hidden_dim=hidden_dim)
        specs = [InputSpec(name='x_seq', shape=(self.lookback, 1), type='seq'), InputSpec(name='x_spatial', shape=(1, self.n_scales, self.lookback), type='spatial'), InputSpec(name='x_graph', shape=(8,), type='graph')]
        challenger = MultiModalRankNet(specs=specs, hidden_dim=hidden_dim, vit_heads=vit_heads, gnn_layers=gnn_layers)

        if not skip_train:
            # 1. Dataset Prep
            cache_path = "data/train_dataset.pt"
            if os.path.exists(cache_path):
                print(f"📦 LOADING CACHED TRAINING FEATURES: {cache_path}...")
                from datetime import datetime as dt_class
                torch.serialization.add_safe_globals([MultiModalBatch, dt_class])
                train_dataset = torch.load(cache_path, weights_only=True)
                if train_dataset.data['x_spatial'].shape[2] != self.n_scales:
                    print(f"⚠️ Cache config mismatch. RE-COLLECTING..."); train_dataset = self._collect_training_data(train_start, train_end); torch.save(train_dataset, cache_path)
            else:
                train_dataset = self._collect_training_data(train_start, train_end); torch.save(train_dataset, cache_path)
            
            if device.type == 'cuda':
                print(f"🚀 MOVING ENTIRE DATASET TO GPU VRAM (~4GB)..."); train_dataset = train_dataset.to(device)

            print(f"🛠️ Training Models on {len(train_dataset.labels)} samples using {device}...")
            patience = self.config['model_pipeline']['training'].get('early_stopping_patience', 3)
            epochs, lr = self.config['model_pipeline']['training']['epochs'], self.config['model_pipeline']['training']['lr']

            # 2. Train Champion (MLP)
            print("🏆 Training Champion (MLP)...")
            champion.fit((train_dataset.data['x_spatial'][:, 0, :, -1], train_dataset.labels), epochs=epochs, lr=lr, device=device, batch_size=batch_size, patience=patience)
            
            # 3. Train Challenger (Multi-Modal)
            print("🚀 Training Challenger (Multi-Modal)...")
            challenger.fit(train_dataset, epochs=epochs, lr=lr, device=device, batch_size=batch_size, patience=patience)
            
            # PERSISTENCE: Save the trained model
            os.makedirs("models", exist_ok=True)
            challenger.export("models/challenger_v1.pt")
            # Also export champion for OOS comparison
            torch.save(champion.state_dict(), "models/champion_baseline.pt")
            print("✅ Training complete. Models exported to models/")
        else:
            print("⏭️ SKIPPING TRAINING. Loading existing models from models/...")
            try:
                # Challenger is TorchScript
                challenger = torch.jit.load("models/challenger_v1.pt")
                challenger.to(device)
                
                # Champion is state_dict (or we can just leave it as is if we don't have it)
                if os.path.exists("models/champion_baseline.pt"):
                    champion.load_state_dict(torch.load("models/champion_baseline.pt", weights_only=True))
                champion.to(device)
                print("✅ Models loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading models: {e}. Please run without --eval-only first.")
                return

        # 4. EVALUATION PHASE
        print(f"🧪 Evaluating on OOS regime ({test_start.date()} -> {test_end.date()})...")
        steps = self.universe.walk_forward(universe=self.tickers, start_date=test_start, end_date=test_end, stride=21, lookback=self.lookback, horizon=5)
        results = []
        for step in steps:
            batch = step['batch'].to(device) if device.type == 'cuda' else step['batch']
            with torch.no_grad():
                # Inference
                if isinstance(champion, torch.jit.ScriptModule):
                    # Should not happen here but safety first
                    champ_scores = champion(batch.data['x_spatial'][:, 0, :, -1]).squeeze().cpu().numpy()
                else:
                    champion.to(device)
                    champ_scores = champion(batch.data['x_spatial'][:, 0, :, -1]).squeeze().cpu().numpy()
                
                if hasattr(challenger, 'predict_dataset'):
                    challenger_scores = challenger.predict_dataset(batch, batch_size=batch_size).squeeze().cpu().numpy()
                else:
                    # Loaded via jit.load
                    # Handle MultiModal input dict
                    inputs = {k: v.to(device) for k, v in batch.data.items()}
                    challenger_scores = challenger(inputs).squeeze().cpu().numpy()
                    
            target = batch.labels.cpu().numpy()
            c_ic = 0 if np.std(champ_scores) < 1e-6 or np.std(target) < 1e-6 else spearmanr(champ_scores, target)[0]
            ch_ic = 0 if np.std(challenger_scores) < 1e-6 or np.std(target) < 1e-6 else spearmanr(challenger_scores, target)[0]
            results.append({'date': step['date'], 'champion_ic': c_ic if not np.isnan(c_ic) else 0, 'challenger_ic': ch_ic if not np.isnan(ch_ic) else 0})
            
        df_results = pd.DataFrame(results)
        df_results.to_csv("data/backtest_results.csv", index=False)
        print("\n--- Production Backtest Summary ---")
        print(f"Champion Avg IC: {df_results['champion_ic'].mean():.4f}\nChallenger Avg IC: {df_results['challenger_ic'].mean():.4f}")
        print(f"Challenger Win Rate: {(df_results['challenger_ic'] > df_results['champion_ic']).mean()*100:.1f}%\n💾 Results saved to data/backtest_results.csv")
        return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UQTS-2026 Production Backtest Orchestrator")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion")
    parser.add_argument("--train", action="store_true", help="Run model training and comparison")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and load existing model for evaluation")
    args = parser.parse_args()
    
    orchestrator = BacktestOrchestrator()
    
    if args.ingest: orchestrator.run_ingestion()
    
    # Logic:
    # 1. If --eval-only is passed, we skip training.
    # 2. If --train is passed, we run training.
    # 3. If nothing is passed, we do everything.
    
    do_train = args.train or (not args.ingest and not args.eval_only)
    skip_train = args.eval_only
    
    if do_train or skip_train:
        orchestrator.run_comparison(
            datetime(2018, 1, 1), 
            datetime(2022, 12, 31), 
            datetime(2023, 1, 1), 
            datetime.now(),
            skip_train=skip_train
        )
