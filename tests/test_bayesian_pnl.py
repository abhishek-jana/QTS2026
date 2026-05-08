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

print("🚀 FAST OOS SIMULATION BASED ON ACTUAL IC 🚀")
print(f"Total Trading Days: {len(df)}")

for idx, row in df.iterrows():
    ic = row['challenger_ic']
    
    # Bayesian Likelihood (same as meta_controller.py)
    likelihood_valid = 1 / (1 + np.exp(-5 * (ic - 0.01)))
    likelihood_invalid = 1 - likelihood_valid
    
    # Bayesian Update
    marginal = (likelihood_valid * belief) + (likelihood_invalid * (1 - belief))
    if marginal > 0:
        belief = (likelihood_valid * belief) / marginal
        
    # Confidence Floor (same as meta_controller)
    belief = max(0.05, min(0.95, belief))
    
    # Trading Logic
    if belief > threshold:
        active_days += 1
        # Realistic daily drift for this IC
        daily_drift = (ic * 0.05) + np.random.normal(0, 0.002) # P&L based on correlation
        capital *= (1.0 + daily_drift)

print("\n" + "="*40)
print("🏁 SIMULATION COMPLETE 🏁")
print(f"Final Account Value: ${capital:,.2f}")
print(f"Total Return: {((capital/starting_capital)-1)*100:.2f}%")
print(f"Final Bayesian Belief: {belief*100:.1f}%")
print(f"Days Actively Traded: {active_days} / {len(df)}")
print("="*40 + "\n")
