import torch
import torch.optim as optim
from research_lab.alpha_ranker import RankNet, PairwiseRankLoss

def test_ranknet_convergence():
    """
    Verify RankNet converges using the new high-level fit() interface.
    """
    torch.manual_seed(42)
    input_dim = 10
    model = RankNet(input_dim)
    criterion = PairwiseRankLoss()
    
    # Simple synthetic data
    i_features = torch.randn(100, input_dim) + 2.0
    j_features = torch.randn(100, input_dim) - 2.0
    target = torch.ones(100, 1)
    
    # Prepare X, y for the new fit() interface
    X = torch.cat([i_features, j_features], dim=0)
    y = torch.cat([torch.ones(100, 1), torch.zeros(100, 1)], dim=0)

    # Initial loss check
    model.eval()
    with torch.no_grad():
        s_i_init = model(i_features)
        s_j_init = model(j_features)
        initial_loss = criterion(s_i_init, s_j_init, target).item()
    
    # High-level fit
    model.fit(X, y, epochs=100, lr=0.01)
    
    # Final loss check
    model.eval()
    with torch.no_grad():
        s_i_final = model(i_features)
        s_j_final = model(j_features)
        final_loss = criterion(s_i_final, s_j_final, target).item()
        
    print(f"Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss
    assert final_loss < 0.2 # Convergence check

def test_ranknet_predict_export(tmp_path):
    """
    Verify predict() and export() methods.
    """
    input_dim = 10
    model = RankNet(input_dim)
    X = torch.randn(5, input_dim)
    
    # Test predict
    scores = model.predict(X)
    assert scores.shape == (5, 1)
    
    # Test export
    export_path = tmp_path / "model.pt"
    model.export(str(export_path))
    assert export_path.exists()
    
    # Verify we can load it (optional but good)
    loaded_model = torch.jit.load(str(export_path))
    loaded_scores = loaded_model(X)
    assert torch.allclose(scores, loaded_scores)

if __name__ == "__main__":
    test_ranknet_convergence()
