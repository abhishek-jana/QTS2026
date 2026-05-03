import torch
import torch.optim as optim
from research_lab.alpha_ranker import RankNet, PairwiseRankLoss

def test_ranknet_convergence():
    """
    TDD: Verify RankNet converges on a simple pairwise task.
    Task: i is better than j if feature[0] of i > feature[0] of j.
    """
    torch.manual_seed(42)
    input_dim = 10
    model = RankNet(input_dim)
    criterion = PairwiseRankLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Simple synthetic data
    # item_i: [1, 0, ...]
    # item_j: [0, 1, ...]
    # Target: i > j (target = 1)
    i_features = torch.randn(100, input_dim) + 2.0
    j_features = torch.randn(100, input_dim) - 2.0
    target = torch.ones(100, 1)
    
    initial_loss = 0
    final_loss = 0
    
    for epoch in range(50):
        optimizer.zero_grad()
        s_i = model(i_features)
        s_j = model(j_features)
        loss = criterion(s_i, s_j, target)
        loss.backward()
        optimizer.step()
        
        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        
    print(f"Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss
    assert final_loss < 0.2 # Convergence check

if __name__ == "__main__":
    test_ranknet_convergence()
