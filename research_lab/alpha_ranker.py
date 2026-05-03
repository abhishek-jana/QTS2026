import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

class LightweightViT(nn.Module):
    """
    Custom Lightweight Vision Transformer for Market Spectrograms.
    Avoids heavy library dependencies and large image upsampling.
    """
    def __init__(self, img_size=(8, 63), patch_size=(8, 9), in_chans=1, embed_dim=64, depth=2, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.patch_embed = nn.Linear(patch_size[0] * patch_size[1] * in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (N, 1, H, W)
        p1, p2 = self.patch_size
        # Patching: (N, C, H, W) -> (N, NumPatches, PatchDim)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2)
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        return x[:, 0] # Return CLS token embedding

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
        High-level training interface for RankNet models.
        Supports MultiModalDataset or raw tensors.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = PairwiseRankLoss()

        if hasattr(dataset, '__len__'):
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
            
            for epoch in range(epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    
                    if isinstance(batch, dict):
                        x_seq = batch['x_seq']
                        x_spatial = batch['x_spatial']
                        y = batch['y']
                        
                        n = y.size(0)
                        if n < 2: continue
                        
                        idx_i = torch.randperm(n)
                        idx_j = torch.randperm(n)
                        
                        s_i = self.forward_multi(x_seq[idx_i], x_spatial[idx_i])
                        s_j = self.forward_multi(x_seq[idx_j], x_spatial[idx_j])
                        target = torch.sign(y[idx_i] - y[idx_j])
                    else:
                        X, y = batch
                        n = y.size(0)
                        if n < 2: continue
                        idx_i = torch.randperm(n)
                        idx_j = torch.randperm(n)
                        s_i = self.forward(X[idx_i])
                        s_j = self.forward(X[idx_j])
                        target = torch.sign(y[idx_i] - y[idx_j])

                    loss = criterion(s_i, s_j, target.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

    def forward_multi(self, x_seq, x_spatial):
        return self.forward(x_seq)

    @torch.jit.ignore
    def predict_dataset(self, dataset):
        self.eval()
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        scores = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    scores.append(self.forward_multi(batch['x_seq'], batch['x_spatial']))
                else:
                    scores.append(self.forward(batch[0]))
        return torch.cat(scores)

    @torch.jit.ignore
    def export(self, path: str):
        self.eval()
        print("Export skipped: MultiModal tracing required.")

class MultiModalRankNet(RankNet):
    """
    UQTS-2026 Multi-Modal RankNet (LSTM + ViT).
    """
    def __init__(self, scales: int = 8, lookback: int = 63, hidden_dim: int = 64):
        super(MultiModalRankNet, self).__init__(input_dim=hidden_dim*2, hidden_dim=hidden_dim)
        
        # 1. Sequential Encoder (LSTM)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        # 2. Spatial Encoder (Custom Lightweight ViT)
        self.vit = LightweightViT(img_size=(scales, lookback), patch_size=(scales, 9), embed_dim=hidden_dim)

    def forward_multi(self, x_seq: torch.Tensor, x_spatial: torch.Tensor) -> torch.Tensor:
        # x_seq: (N, T, 1)
        # x_spatial: (N, 1, Scales, T)
        
        # 1. Sequential Embedding
        _, (h_n, _) = self.lstm(x_seq)
        seq_emb = h_n[-1] 
        
        # 2. Spatial Embedding
        spatial_emb = self.vit(x_spatial) 
        
        # 3. Late Fusion
        fusion_emb = torch.cat([seq_emb, spatial_emb], dim=1) 
        
        # 4. Ranking Head
        return self.model(fusion_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class PairwiseRankLoss(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super(PairwiseRankLoss, self).__init__()
        self.sigma = sigma

    def forward(self, s_i: torch.Tensor, s_j: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = s_i - s_j
        p_bar = (1 + target) / 2
        loss = -p_bar * self.sigma * diff + torch.log(1 + torch.exp(self.sigma * diff))
        return loss.mean()
