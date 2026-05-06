import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

@dataclass
class InputSpec:
    name: str
    shape: Tuple[int, ...]
    type: str  # 'seq', 'spatial', or 'graph'

class LightweightViT(nn.Module):
    def __init__(self, img_size=(32, 63), patch_size=(32, 9), in_chans=1, embed_dim=64, depth=2, num_heads=4):
        super().__init__()
        self.img_size, self.patch_size, self.in_chans = img_size, patch_size, in_chans
        hp, wp = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.patch_dim = patch_size[0] * patch_size[1] * in_chans
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, hp * wp + 1, embed_dim))
        el = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p1, p2, C = self.patch_size[0], self.patch_size[1], self.in_chans
        hp, wp = self.img_size[0] // p1, self.img_size[1] // p2
        x = x.view(B, C, hp, p1, wp, p2).permute(0, 2, 4, 3, 5, 1).contiguous().view(B, hp * wp, -1)
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) + self.pos_embed
        return self.norm(self.transformer(x)[:, 0])

class LightweightGNN(nn.Module):
    def __init__(self, in_features: int, embed_dim: int, depth: int = 2, num_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim)
        el = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.gnn_encoder = nn.TransformerEncoder(el, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).unsqueeze(1).permute(1, 0, 2)
        x = self.gnn_encoder(x).permute(1, 0, 2).squeeze(1)
        return self.norm(x)

class RankNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                                   nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.model(x)

    @torch.jit.ignore
    def fit(self, dataset, epochs: int = 50, lr: float = 0.01, verbose: bool = True, 
            device: torch.device = torch.device('cpu'), batch_size: int = 1024, patience: int = 5):
        self.to(device); self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = PairwiseRankLoss()
        
        # MIXED PRECISION: Target Tensor Cores for 2x speedup
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        if hasattr(dataset, 'data') and hasattr(dataset, 'labels'):
            ns, bl, wait = len(dataset.labels), float('inf'), 0
            nb = ns // batch_size
            log_interval = max(1, nb // 10)
            for epoch in range(epochs):
                tl, indices = 0, torch.randperm(ns, device=device)
                for b in range(nb):
                    optimizer.zero_grad()
                    idx = indices[b*batch_size : (b+1)*batch_size]
                    y_b = dataset.labels[idx]
                    ii, jj = torch.randperm(batch_size, device=device), torch.randperm(batch_size, device=device)
                    
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        in_i = {k: v[idx][ii] for k, v in dataset.data.items()}
                        in_j = {k: v[idx][jj] for k, v in dataset.data.items()}
                        s_i, s_j = self.forward(in_i), self.forward(in_j)
                        loss = criterion(s_i, s_j, torch.sign(y_b[ii] - y_b[jj]).unsqueeze(1))
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    tl += loss.item()
                    if verbose and b % log_interval == 0:
                        print(f"      > Batch {b}/{nb} | Loss: {loss.item():.4f}", end='\r')
                al = tl / max(1, nb)
                if verbose: print(f"   [Epoch {epoch+1}/{epochs}] Avg Loss: {al:.6f}")
                if al < bl: bl, wait = al, 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose: print(f"   🛑 Early stopping at epoch {epoch+1}"); break
            return
        if isinstance(dataset, tuple):
            X_all, y_all = dataset
            ns = len(y_all)
            nb = ns // batch_size
            bl, wait = float('inf'), 0
            for epoch in range(epochs):
                tl, indices = 0, torch.randperm(ns, device=device)
                for b in range(nb):
                    optimizer.zero_grad()
                    idx = indices[b*batch_size : (b+1)*batch_size]
                    X, y = X_all[idx].to(device), y_all[idx].to(device)
                    ii, jj = torch.randperm(X.size(0)), torch.randperm(X.size(0))
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        loss = criterion(self.forward(X[ii]), self.forward(X[jj]), torch.sign(y[ii] - y[jj]).unsqueeze(1))
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    tl += loss.item()
                al = tl / max(1, nb)
                if verbose: print(f"   [Epoch {epoch+1}/{epochs}] Avg Loss: {al:.6f}")
                if al < bl: bl, wait = al, 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose: print(f"   🛑 Early stopping at epoch {epoch+1}"); break

    @torch.jit.ignore
    def predict_dataset(self, dataset, batch_size: int = 1024):
        self.eval(); device = next(self.parameters()).device
        if hasattr(dataset, 'data'):
            ns, scores = len(dataset.labels), []
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                for i in range(0, ns, batch_size):
                    e = min(i + batch_size, ns); inputs = {k: v[i:e].to(device) for k, v in dataset.data.items()}
                    scores.append(self.forward(inputs).cpu())
            return torch.cat(scores)
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False); scores = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            for batch in loader: scores.append(self.forward(batch[0].to(device)).cpu())
        return torch.cat(scores)

    @torch.jit.ignore
    def export(self, path: str):
        self.eval()
        device = next(self.parameters()).device
        if hasattr(self, 'specs'):
            # FIX: Ensure example inputs are on the same device as the model
            example_inputs = {spec.name: torch.randn(1, *spec.shape).to(device) for spec in self.specs}
            traced_model = torch.jit.trace(self, (example_inputs,), check_trace=False)
        else:
            example_x = torch.randn(1, self.model[0].in_features).to(device)
            traced_model = torch.jit.trace(self, (example_x,), check_trace=False)
        traced_model.save(path); print(f"✅ RankNet serialized to {path}.")

class MultiModalRankNet(RankNet):
    def __init__(self, specs: List[InputSpec] = None, hidden_dim: int = 64, **kwargs):
        if specs is None:
            scales, lookback = kwargs.get('scales', 32), kwargs.get('lookback', 63)
            specs = [InputSpec(name='x_seq', shape=(lookback, 1), type='seq'), InputSpec(name='x_spatial', shape=(1, scales, lookback), type='spatial')]
        super(MultiModalRankNet, self).__init__(input_dim=hidden_dim * len(specs), hidden_dim=hidden_dim)
        self.specs, self.hidden_dim, self.encoders = specs, hidden_dim, nn.ModuleDict()
        vh, gl = kwargs.get('vit_heads', 4), kwargs.get('gnn_layers', 2)
        for spec in specs:
            if spec.type == 'seq': self.encoders[spec.name] = nn.LSTM(input_size=spec.shape[-1], hidden_size=hidden_dim, num_layers=2, batch_first=True)
            elif spec.type == 'spatial': self.encoders[spec.name] = LightweightViT(img_size=spec.shape[1:], patch_size=(spec.shape[1], 9), in_chans=spec.shape[0], embed_dim=hidden_dim, num_heads=vh)
            elif spec.type == 'graph': self.encoders[spec.name] = LightweightGNN(in_features=spec.shape[0], embed_dim=hidden_dim, depth=gl)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = []
        for spec in self.specs:
            x = inputs[spec.name]
            if spec.type == 'seq': _, (h_n, _) = self.encoders[spec.name](x); embs.append(h_n[-1])
            elif spec.type == 'spatial': embs.append(self.encoders[spec.name](x))
            elif spec.type == 'graph': embs.append(self.encoders[spec.name](x))
        return self.model(torch.cat(embs, dim=1))

class PairwiseRankLoss(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super(PairwiseRankLoss, self).__init__()
        self.sigma = sigma
    def forward(self, s_i, s_j, target):
        diff = s_i - s_j
        p_bar = (1 + target) / 2
        return (-p_bar * self.sigma * diff + torch.log(1 + torch.exp(self.sigma * diff))).mean()
