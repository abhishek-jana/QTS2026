import pandas as pd
import numpy as np

# Load the actual backtest results
df = pd.read_csv("data/backtest_results.csv")

# Simulation parameters
starting_capital = 100000.0
capital = starting_capital
belief = 0.50 # Prior
threshold = 0.65
active_days = 0
gross_exposure = 0.60 # 12 positions (6 Long, 6 Short) * 5% per position = 60%

print("🚀 FAST OOS SIMULATION (OPTION B: 60% EXPOSURE) 🚀")

np.random.seed(42) # For reproducible estimate

for idx, row in df.iterrows():
    ic = row['challenger_ic']
    
    # Bayesian Likelihood
    likelihood_valid = 1 / (1 + np.exp(-5 * (ic - 0.01)))
    likelihood_invalid = 1 - likelihood_valid
    
    # Bayesian Update
    marginal = (likelihood_valid * belief) + (likelihood_invalid * (1 - belief))
    if marginal > 0:
        belief = (likelihood_valid * belief) / marginal
        
    belief = max(0.05, min(0.95, belief))
    
    # Trading Logic
    if belief > threshold:
        active_days += 1
        # Realistic daily drift scaled by our 60% Gross Exposure
        daily_drift = ((ic * 0.05) + np.random.normal(0, 0.002)) * gross_exposure
        capital *= (1.0 + daily_drift)

print("\n" + "="*40)
print("🏁 MACRO PROJECTION COMPLETE 🏁")
print(f"Final Account Value: ${capital:,.2f}")
print(f"Total Return: {((capital/starting_capital)-1)*100:.2f}%")
print(f"Final Bayesian Belief: {belief*100:.1f}%")
print(f"Days Actively Traded: {active_days} / {len(df)}")
print("="*40 + "\n")
