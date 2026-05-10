import numpy as np

# Simulate typical RankNet outputs
scores = np.array([2.5, 1.8, 1.2, 0.5, -0.1])
exp_scores = np.exp(scores - np.max(scores))
weights = exp_scores / np.sum(exp_scores)

print("Scores:", scores)
print("Weights:", np.round(weights, 3))
print("Max weight %:", np.max(weights) * 100)

scores_spiky = np.array([5.5, 1.2, 0.8, 0.5, 0.1])
exp_scores_s = np.exp(scores_spiky - np.max(scores_spiky))
weights_s = exp_scores_s / np.sum(exp_scores_s)
print("\nSpiky Scores:", scores_spiky)
print("Spiky Weights:", np.round(weights_s, 3))
print("Max weight %:", np.max(weights_s) * 100)

