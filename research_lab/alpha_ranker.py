import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LightweightViT(nn.Module):
    """
    Custom Lightweight Vision Transformer for Market Spectrograms.
    Optimized for TorchScript compatibility using native torch ops.
    """
    def __init__(self, img_size=(8, 63), patch_size=(8, 9), in_chans=1, embed_dim=64, depth=2, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        h_patches = img_size[0] // patch_size[0]
        w_patches = img_size[1] // patch_size[1]
        num_patches = h_patches * w_patches
        
        self.patch_dim = patch_size[0] * patch_size[1] * in_chans
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 8, 63)
        B = x.shape[0]
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        C = self.in_chans
        
        # Native Patching
        # (B, C, H, W) -> (B, C, H//p1, p1, W//p2, p2)
        H_p = self.img_size[0] // p1
        W_p = self.img_size[1] // p2
        
        x = x.view(B, C, H_p, p1, W_p, p2)
        # -> (B, H_p, W_p, p1, p2, C)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        # -> (B, N_patches, Patch_dim)
        x = x.view(B, H_p * W_p, -1)
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        x = self.norm(x)
        
        return x[:, 0]

class RankNet(nn.Module):
    """
    Base RankNet.
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
    def fit(self, dataset, epochs: int = 50, lr: float = 0.01):
        """
        High-level training interface.
        Supports MultiModalBatch, TensorDataset, or (X, y) tuple.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = PairwiseRankLoss()

        # Handle (X, y) tuple
        if isinstance(dataset, tuple) and len(dataset) == 2:
            from torch.utils.data import TensorDataset
            dataset = TensorDataset(*dataset)

        if hasattr(dataset, '__len__'):
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
            
            for epoch in range(epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    if isinstance(batch, dict):
                        idx_i = torch.randperm(batch['y'].size(0))
                        idx_j = torch.randperm(batch['y'].size(0))
                        s_i = self.forward(batch['x_seq'][idx_i], batch['x_spatial'][idx_i])
                        s_j = self.forward(batch['x_seq'][idx_j], batch['x_spatial'][idx_j])
                        target = torch.sign(batch['y'][idx_i] - batch['y'][idx_j])
                    else:
                        X, y = batch
                        idx_i = torch.randperm(y.size(0))
                        idx_j = torch.randperm(y.size(0))
                        s_i = self.forward(X[idx_i])
                        s_j = self.forward(X[idx_j])
                        target = torch.sign(y[idx_i] - y[idx_j])

                    loss = criterion(s_i, s_j, target.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

    @torch.jit.ignore
    def predict_dataset(self, dataset):
        self.eval()
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        scores = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    scores.append(self.forward(batch['x_seq'], batch['x_spatial']))
                else:
                    scores.append(self.forward(batch[0]))
        return torch.cat(scores)

    @torch.jit.ignore
    def export(self, path: str):
        """
        Serializes the model via Tracing for production execution.
        """
        self.eval()
        example_seq = torch.randn(1, 63, 1)
        example_spatial = torch.randn(1, 1, 8, 63)
        
        # Trace with check_trace=False to handle potential Transformer graph subtleties
        traced_model = torch.jit.trace(self, (example_seq, example_spatial), check_trace=False)
        traced_model.save(path)
        print(f"✅ Multi-Modal RankNet serialized to {path} via Tracing.")

class MultiModalRankNet(RankNet):
    """
    UQTS-2026 Multi-Modal RankNet (LSTM + ViT).
    """
    def __init__(self, scales: int = 8, lookback: int = 63, hidden_dim: int = 64):
        super(MultiModalRankNet, self).__init__(input_dim=hidden_dim*2, hidden_dim=hidden_dim)
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.vit = LightweightViT(img_size=(scales, lookback), patch_size=(scales, 9), embed_dim=hidden_dim)

    def forward(self, x_seq: torch.Tensor, x_spatial: torch.Tensor) -> torch.Tensor:
        # 1. Sequential Embedding
        _, (h_n, _) = self.lstm(x_seq)
        seq_emb = h_n[-1] 
        
        # 2. Spatial Embedding
        spatial_emb = self.vit(x_spatial) 
        
        # 3. Late Fusion
        fusion_emb = torch.cat([seq_emb, spatial_emb], dim=1) 
        
        # 4. Ranking Head
        return self.model(fusion_emb)

class PairwiseRankLoss(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super(PairwiseRankLoss, self).__init__()
        self.sigma = sigma

    def forward(self, s_i: torch.Tensor, s_j: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = s_i - s_j
        p_bar = (1 + target) / 2
        loss = -p_bar * self.sigma * diff + torch.log(1 + torch.exp(self.sigma * diff))
        return loss.mean()
