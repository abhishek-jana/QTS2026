import torch
import torch.nn as nn
import torch.optim as optim

class RankNet(nn.Module):
    """
    Standard RankNet architecture for Learning to Rank (LTR).
    Maps high-dimensional wavelet features to a single scalar score.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.jit.ignore
    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 50, lr: float = 0.01):
        """
        High-level training interface that encapsulates the optimizer,
        loss calculation, and training loop.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = PairwiseRankLoss()

        n_samples = X.size(0)
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Generate random pairs for pairwise training
            # Using more pairs than samples to ensure better coverage
            n_pairs = n_samples * 2
            idx_i = torch.randint(0, n_samples, (n_pairs,))
            idx_j = torch.randint(0, n_samples, (n_pairs,))
            
            s_i = self.forward(X[idx_i])
            s_j = self.forward(X[idx_j])
            
            # target: 1 if i > j, -1 if j > i, 0 if equal
            target = torch.sign(y[idx_i] - y[idx_j])
            
            loss = criterion(s_i, s_j, target)
            loss.backward()
            optimizer.step()

    @torch.jit.ignore
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Standard inference interface.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    @torch.jit.ignore
    def export(self, path: str):
        """
        Serialization to TorchScript for C++ inference.
        """
        self.export_to_torchscript(path)

    def export_to_torchscript(self, path: str):
        """
        Serializes the model to TorchScript for C++ inference.
        """
        self.eval()
        scripted_model = torch.jit.script(self)
        scripted_model.save(path)
        print(f"Model exported to {path}")

class PairwiseRankLoss(nn.Module):
    """
    Pairwise RankNet Loss.
    Encourages s_i > s_j when item i is ranked higher than item j.
    """
    def __init__(self, sigma: float = 1.0):
        super(PairwiseRankLoss, self).__init__()
        self.sigma = sigma

    def forward(self, s_i: torch.Tensor, s_j: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        target: 1 if i should be higher than j, 0 if equal, -1 if j higher.
        """
        # P_{ij} = 1 / (1 + exp(-sigma * (s_i - s_j)))
        # Target probability P_bar = (1 + target) / 2
        # Cross-entropy loss
        diff = s_i - s_j
        p_bar = (1 + target) / 2
        loss = -p_bar * self.sigma * diff + torch.log(1 + torch.exp(self.sigma * diff))
        return loss.mean()
