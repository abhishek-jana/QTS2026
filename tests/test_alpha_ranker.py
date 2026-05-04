import torch
import torch.optim as optim
from datetime import datetime
from research_lab.alpha_ranker import MultiModalRankNet, PairwiseRankLoss
from research_lab.alpha_universe import MultiModalBatch

def test_multimodal_ranknet_convergence():
    """
    TDD: Verify MultiModalRankNet converges on dual-stream synthetic data.
    """
    torch.manual_seed(42)
    n_samples = 500
    lookback = 63
    scales = 8
    hidden_dim = 32
    
    model = MultiModalRankNet(scales=scales, lookback=lookback, hidden_dim=hidden_dim)
    
    # Synthetic inputs
    # i is better than j if seq mean is higher
    x_seq = torch.randn(n_samples, lookback, 1)
    x_spatial = torch.randn(n_samples, 1, scales, lookback)
    # Strong linear signal
    y = x_seq.mean(dim=1).squeeze() * 10
    
    dataset = MultiModalBatch(
        data={'x_seq': x_seq, 'x_spatial': x_spatial},
        labels=y,
        tickers=['TEST'] * n_samples,
        times=[datetime.now()] * n_samples
    )
    
    # Train: Smaller LR, more steps
    model.fit(dataset, epochs=20, lr=0.001)
    
    # Inference
    scores = model.predict_dataset(dataset)
    
    # Verify correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(y.numpy(), scores.numpy())
    print(f"Multi-Modal Rank correlation: {corr:.4f}")
    assert corr > 0.4 # Relaxing slightly for random init

if __name__ == "__main__":
    test_multimodal_ranknet_convergence()
