import torch
import os
from research_lab.alpha_ranker import RankNet

def test_torchscript_export_consistency():
    """
    TDD: Verify that exported TorchScript model produces same output as PyTorch.
    """
    input_dim = 128
    model = RankNet(input_dim)
    model.eval()
    
    # Export
    export_path = "test_models/model.pt"
    model.export(export_path)
    
    # Load back
    loaded_model = torch.jit.load(export_path)
    
    # Compare outputs
    test_input = torch.randn(10, input_dim)
    with torch.no_grad():
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        
    os.remove(export_path)
    assert torch.allclose(original_output, loaded_output, atol=1e-6)
    print("TorchScript Consistency: PASS")

if __name__ == "__main__":
    test_torchscript_export_consistency()
