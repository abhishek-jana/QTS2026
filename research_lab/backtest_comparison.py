import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from research_lab.alpha_universe import AlphaUniverse
from research_lab.alpha_ranker import RankNet, MultiModalRankNet
from research_lab.plugins.core_plugins import SequentialPlugin, SpatialPlugin
from research_lab.real_data_ingestor import YFinanceIngestor
from scipy.stats import spearmanr

current_date = datetime.now().strftime('%Y-%m-%d')

class BacktestOrchestrator:
    """
    Executes and compares 'Champion' vs 'Challenger' models on real data.
    """
    def __init__(self):
        # 1. Setup Universe with Expanded tech basket
        self.tickers = ['AAPL', 'MSFT', 'GOOG', 'SPY', 'AMZN', 'NFLX', 'META', 'NVDA']
        self.plugins = [SequentialPlugin(), SpatialPlugin()]
        self.universe = AlphaUniverse(plugins=self.plugins)
        
        ingestor = YFinanceIngestor(self.universe.engine)
        # 4-year window for Training + OOS Evaluation
        ingestor.ingest_universe(self.tickers, '2022-01-01', current_date)

    def run_comparison(self, train_start: datetime, train_end: datetime, test_start: datetime, test_end: datetime):
        """
        Trains models on IS data and evaluates on OOS data.
        """
        print(f"📊 INITIATING PRODUCTION TRIAL...")
        print(f"TRAIN: {train_start.date()} -> {train_end.date()}")
        print(f"TEST:  {test_start.date()} -> {test_end.date()}")
        
        # 1. Initialize Models
        champion = RankNet(input_dim=8)
        challenger = MultiModalRankNet(scales=8, lookback=63)
        
        # 2. TRAINING PHASE (In-Sample)
        print("🛠️ Training Models on historical regime...")
        train_batch = self.universe.snapshot(as_of=train_end, tickers=self.tickers, lookback=63)
        if train_batch:
            # Train Champion (MLP)
            # Flatten spatial energy for MLP input
            X_champ = train_batch.data['x_spatial'][:, 0, :, -1]
            champion.fit((X_champ, train_batch.labels), epochs=5)
            
            # Train Challenger (Multi-Modal)
            challenger.fit(train_batch, epochs=5)
            print("✅ Training complete.")

        # 3. EVALUATION PHASE (Out-of-Sample - 2026)
        steps = self.universe.walk_forward(
            universe=self.tickers, 
            start_date=test_start, 
            end_date=test_end, 
            stride=7, # Weekly resolution for 2026 evaluation
            lookback=63,
            horizon=5
        )
        
        results = []
        for step in steps:
            batch = step['batch']
            date = step['date']
            
            # Champion Inference
            champ_input = batch.data['x_spatial'][:, 0, :, -1] 
            with torch.no_grad():
                champ_scores = champion(champ_input).squeeze().numpy()
            
            # Challenger Inference
            with torch.no_grad():
                challenger_scores = challenger.predict_dataset(batch).squeeze().numpy()
                
            # Metrics
            target = batch.labels.numpy()
            champ_ic, _ = spearmanr(champ_scores, target)
            chall_ic, _ = spearmanr(challenger_scores, target)
            
            results.append({
                'date': date,
                'champion_ic': champ_ic,
                'challenger_ic': chall_ic
            })
            
        df_results = pd.DataFrame(results)
        print("\n--- 2026 OOS Backtest Summary ---")
        print(f"Champion Avg IC: {df_results['champion_ic'].mean():.4f}")
        print(f"Challenger Avg IC: {df_results['challenger_ic'].mean():.4f}")
        
        win_rate = (df_results['challenger_ic'] > df_results['champion_ic']).mean()
        print(f"Challenger Win Rate vs. Benchmark: {win_rate*100:.1f}%")
        
        return df_results

if __name__ == "__main__":
    orchestrator = BacktestOrchestrator()
    # 2022-2025 Training
    train_start = datetime(2022, 1, 1)
    train_end = datetime(2025, 12, 31)
    # 2026 Out-of-Sample Evaluation
    test_start = datetime(2026, 1, 1)
    test_end = datetime(2026, 5, 3)
    orchestrator.run_comparison(train_start, train_end, test_start, test_end)
