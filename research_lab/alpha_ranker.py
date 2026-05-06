import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from qts_core.logger import logger

@dataclass
class InputSpec:
    name: str
    shape: Tuple[int, ...]
    type: str

class LightweightViT(nn.Module):
    def __init__(self, img_size=(8, 63), patch_size=(8, 9), in_chans=1, embed_dim=64, depth=2, num_heads=4, dropout=0.1):
        super().__init__()
        # SENIOR FIX: Ensure patch_size is compatible with img_size
        self.img_size, self.patch_size, self.in_chans = img_size, patch_size, in_chans
        hp, wp = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        
        # Guard against zero patches
        hp = max(1, hp); wp = max(1, wp)
        
        self.patch_dim = patch_size[0] * patch_size[1] * in_chans
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, hp * wp + 1, embed_dim))
        el = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(el, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]; p1, p2, C = self.patch_size[0], self.patch_size[1], self.in_chans
        hp, wp = self.img_size[0] // p1, self.img_size[1] // p2
        # Safety for small images
        if hp == 0 or wp == 0:
             x = F.interpolate(x, size=(max(p1, self.img_size[0]), max(p2, self.img_size[1])))
             hp, wp = x.shape[2] // p1, x.shape[3] // p2
        x = x.view(B, C, hp, p1, wp, p2).permute(0, 2, 4, 3, 5, 1).contiguous().view(B, hp * wp, -1)
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) + self.pos_embed
        return self.norm(self.transformer(x)[:, 0])

class LightweightGNN(nn.Module):
    def __init__(self, in_features: int, embed_dim: int, depth: int = 2, num_heads: int = 4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim)
        el = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.gnn_encoder = nn.TransformerEncoder(el, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).unsqueeze(1).permute(1, 0, 2)
        return self.norm(self.gnn_encoder(x).permute(1, 0, 2).squeeze(1))

class RankNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout=0.2):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
                                   nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.model(x)

    @torch.jit.ignore
    def fit(self, dataset, epochs: int = 50, lr: float = 0.01, verbose: bool = True, 
            device: torch.device = torch.device('cpu'), batch_size: int = 1024, patience: int = 5, 
            weight_decay: float = 0.0, val_dataset: Any = None):
        self.to(device); self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = PairwiseRankLoss()
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        if hasattr(dataset, 'data') and hasattr(dataset, 'labels'):
            ns = len(dataset.labels); abs_ = min(batch_size, ns); nb = ns // abs_
            if nb == 0: nb, abs_ = 1, ns
            bl, wait = float('inf'), 0
            for epoch in range(epochs):
                tl, indices = 0, torch.randperm(ns, device=device)
                for b in range(nb):
                    optimizer.zero_grad()
                    idx = indices[b*abs_ : (b+1)*abs_]
                    if len(idx) < 2: continue
                    yb = dataset.labels[idx]; ii, jj = torch.randperm(len(idx), device=device), torch.randperm(len(idx), device=device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        ini = {k: v[idx][ii] for k, v in dataset.data.items()}; inj = {k: v[idx][jj] for k, v in dataset.data.items()}
                        si, sj = self.forward(ini), self.forward(inj)
                        loss = criterion(si, sj, torch.sign(yb[ii] - yb[jj]).unsqueeze(1))
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); tl += loss.item()
                
                scheduler.step()
                al = tl / max(1, nb)
                v_msg = ""
                if val_dataset:
                    self.eval()
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        v_ns = len(val_dataset.labels)
                        v_idx = torch.randperm(v_ns, device=device)[:min(batch_size, v_ns)]
                        if len(v_idx) >= 2:
                            vii, vjj = torch.randperm(len(v_idx), device=device), torch.randperm(len(v_idx), device=device)
                            v_ini = {k: v[v_idx][vii] for k, v in val_dataset.data.items()}
                            v_inj = {k: v[v_idx][vjj] for k, v in val_dataset.data.items()}
                            v_loss = criterion(self.forward(v_ini), self.forward(v_inj), torch.sign(val_dataset.labels[v_idx][vii] - val_dataset.labels[v_idx][vjj]).unsqueeze(1))
                            v_msg = f" | Val Loss: {v_loss.item():.4f}"
                    self.train()

                if verbose: logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {al:.6f}{v_msg}")
                if al < bl: bl, wait = al, 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose: logger.success(f"Early stopping at epoch {epoch+1}"); break
            return

    @torch.jit.ignore
    def predict_dataset(self, dataset, batch_size: int = 1024):
        self.eval(); device = next(self.parameters()).device
        if hasattr(dataset, 'data'):
            ns, scores = len(dataset.labels), []
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                for i in range(0, ns, batch_size):
                    e = min(i + batch_size, ns); inputs = {k: v[i:e].to(device) for k, v in dataset.data.items()}; scores.append(self.forward(inputs).cpu())
            return torch.cat(scores)
        return torch.tensor([])

    @torch.jit.ignore
    def export(self, path: str):
        self.eval(); device = next(self.parameters()).device
        if hasattr(self, 'specs'):
            example_inputs = {spec.name: torch.randn(1, *spec.shape).to(device) for spec in self.specs}
            torch.jit.trace(self, (example_inputs,), check_trace=False).save(path)
        else:
            torch.jit.trace(self, (torch.randn(1, self.model[0].in_features).to(device),), check_trace=False).save(path)
        logger.success(f"RankNet serialized to {path}")

class MultiModalRankNet(RankNet):
    def __init__(self, specs: List[InputSpec] = None, hidden_dim: int = 64, **kwargs):
        if specs is None:
            scales, lookback = kwargs.get('scales', 32), kwargs.get('lookback', 63)
            specs = [InputSpec(name='x_seq', shape=(lookback, 1), type='seq'), InputSpec(name='x_spatial', shape=(1, scales, lookback), type='spatial')]
        dropout = kwargs.get('dropout', 0.3)
        super(MultiModalRankNet, self).__init__(input_dim=hidden_dim * len(specs), hidden_dim=hidden_dim, dropout=dropout)
        self.specs, self.hidden_dim, self.encoders = specs, hidden_dim, nn.ModuleDict()
        vh, gl = kwargs.get('vit_heads', 4), kwargs.get('gnn_layers', 2)
        for spec in specs:
            if spec.type == 'seq': self.encoders[spec.name] = nn.LSTM(input_size=spec.shape[-1], hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
            elif spec.type == 'spatial':
                # FIXED: Dynamically set patch size based on image height (Scales)
                img_h = spec.shape[1]
                self.encoders[spec.name] = LightweightViT(img_size=spec.shape[1:], patch_size=(img_h, 9), in_chans=spec.shape[0], embed_dim=hidden_dim, num_heads=vh, dropout=dropout)
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
    def __init__(self, sigma: float = 1.0): super(PairwiseRankLoss, self).__init__(); self.sigma = sigma
    def forward(self, s_i, s_j, target):
        return (-((1 + target) / 2) * self.sigma * (s_i - s_j) + torch.log(1 + torch.exp(self.sigma * (s_i - s_j)))).mean()
