import torch
import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta
from qts_core.logger import logger
from alpha_factory.strategy_engine import StrategyEngine
from research_lab.data_engine import DataEngine
from alpha_factory.meta_controller import BayesianMetaController

def run_parameter_sensitivity_test():
    # 1. Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Use AI Model via StrategyEngine
    # This demonstrates "using the AI model"
    data_engine = DataEngine(storage_path=config['data_engine']['storage_path'])
    engine = StrategyEngine(data_provider=data_engine, config_path="config.yaml")
    
    logger.info(f"Model loaded: {engine.model}")
    
    # 3. Load Backtest Results for P&L Simulation
    df = pd.read_csv("data/backtest_results.csv")
    
    # 4. Define Parameter Sets to Test
    scenarios = [
        {"name": "Standard (Config)", "threshold": 0.65, "multiplier": 2.0, "window": 10},
        {"name": "Aggressive", "threshold": 0.55, "multiplier": 5.0, "window": 5},
        {"name": "Conservative", "threshold": 0.75, "multiplier": 1.5, "window": 20},
        {"name": "Hyper-Sensitive", "threshold": 0.60, "multiplier": 10.0, "window": 3},
    ]
    
    starting_capital = 100000.0
    results = []

    for scenario in scenarios:
        capital = starting_capital
        belief = config['risk_metacontroller'].get('prior_belief', 0.5)
        threshold = scenario['threshold']
        multiplier = scenario['multiplier']
        window = scenario['window']
        
        ic_history = []
        active_days = 0
        
        for idx, row in df.iterrows():
            ic = row['challenger_ic']
            ic_history.append(ic)
            
            # Simulated MetaController logic
            recent_ic = np.mean(ic_history[-window:])
            
            # Likelihood with custom multiplier
            likelihood_valid = 1 / (1 + np.exp(-multiplier * recent_ic))
            likelihood_invalid = 1 - likelihood_valid
            
            # Bayesian Update
            marginal = (likelihood_valid * belief) + (likelihood_invalid * (1 - belief))
            if marginal > 0:
                belief = (likelihood_valid * belief) / marginal
            
            belief = max(0.05, min(0.95, belief))
            
            # Trading Logic
            if belief > threshold:
                active_days += 1
                # Daily drift proxy (scaling IC to return)
                daily_drift = (ic * 0.05) + np.random.normal(0, 0.002)
                capital *= (1.0 + daily_drift)
                
        results.append({
            "Scenario": scenario['name'],
            "Final Value": capital,
            "Total Return": (capital/starting_capital - 1) * 100,
            "Active Days": active_days,
            "Final Belief": belief
        })

    # 5. Output Comparison
    print("\n" + "="*80)
    print(f"{'Scenario':20} | {'Return %':>10} | {'Active Days':>12} | {'Final Belief':>12}")
    print("-" * 80)
    for res in results:
        print(f"{res['Scenario']:20} | {res['Total Return']:9.2f}% | {res['Active Days']:12} | {res['Final Belief']:11.1%}")
    print("="*80 + "\n")

    # 6. Run a single inference to verify model usage
    logger.info("Running live inference test on current date...")
    try:
        # Pick a date from the dataset range
        test_date = datetime(2023, 6, 1, 16, 0)
        view = engine.get_current_rankings(as_of=test_date)
        if view['status'] == "OK":
            logger.success(f"Inference Successful for {test_date.date()}")
            print("\nTOP 5 TICKERS BY MODEL SCORE:")
            for item in view['ladder'][:5]:
                print(f"{item['ticker']}: {item['score']:.4f} (Energy: {item['energy']:.4f})")
        else:
            logger.warning(f"Inference failed: {view['status']}")
    except Exception as e:
        logger.error(f"Inference Error: {e}")

if __name__ == "__main__":
    run_parameter_sensitivity_test()
