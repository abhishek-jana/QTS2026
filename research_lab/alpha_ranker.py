import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
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
        self.img_size, self.patch_size, self.in_chans = img_size, patch_size, in_chans
        hp, wp = max(1, img_size[0] // patch_size[0]), max(1, img_size[1] // patch_size[1])
        self.patch_dim = patch_size[0] * patch_size[1] * in_chans
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, hp * wp + 1, embed_dim))
        el = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(el, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]; p1, p2, C = self.patch_size[0], self.patch_size[1], self.in_chans
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
    def fit(self, dataset, epochs: int = 50, lr: float = 0.0003, verbose: bool = True, 
            device: torch.device = torch.device('cpu'), batch_size: int = 1024, patience: int = 15, 
            weight_decay: float = 1e-5, val_dataset: Any = None, pointwise_lambda: float = 0.1):
        self.to(device); self.train()
        
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        criterion_pairwise = PairwiseRankLoss()
        criterion_pointwise = nn.MSELoss()
        
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        best_val_loss, wait, best_state = float('inf'), 0, None

        if hasattr(dataset, 'data') and hasattr(dataset, 'labels'):
            ns = len(dataset.labels); abs_ = min(batch_size, ns); nb = ns // abs_
            if nb == 0: nb, abs_ = 1, ns
            for epoch in range(epochs):
                tl, indices = 0, torch.randperm(ns, device=device)
                for b in range(nb):
                    optimizer.zero_grad(); idx = indices[b*abs_ : (b+1)*abs_]
                    if len(idx) < 2: continue
                    yb = dataset.labels[idx]; wb = dataset.sample_weights[idx] if hasattr(dataset, 'sample_weights') and dataset.sample_weights is not None else None
                    ii, jj = torch.randperm(len(idx), device=device), torch.randperm(len(idx), device=device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        ini = {k: v[idx][ii] for k, v in dataset.data.items()}; inj = {k: v[idx][jj] for k, v in dataset.data.items()}
                        si, sj = self.forward(ini), self.forward(inj)
                        target = torch.sign(yb[ii] - yb[jj]).unsqueeze(1)
                        magnitude_weight = torch.abs(yb[ii] - yb[jj]).unsqueeze(1); magnitude_weight = magnitude_weight / (magnitude_weight.mean() + 1e-6)
                        pair_weights = (wb[ii] + wb[jj]).unsqueeze(1) / 2.0 if wb is not None else torch.ones_like(target)
                        loss_pairwise = criterion_pairwise(si, sj, target, weights=pair_weights * magnitude_weight)
                        loss_pointwise = torch.clamp(criterion_pointwise(si, yb[ii].unsqueeze(1)) + criterion_pointwise(sj, yb[jj].unsqueeze(1)), 0, 10.0)
                        loss = loss_pairwise + (pointwise_lambda * loss_pointwise)
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0); scaler.step(optimizer); scaler.update(); tl += loss.item()
                if np.isnan(tl):
                    logger.error(f"💣 CRITICAL: NaN detected at Epoch {epoch+1}. Aborting training."); break
                al = tl / max(1, nb); v_loss_val = None
                if val_dataset:
                    self.eval()
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        v_ns = len(val_dataset.labels); v_idx = torch.randperm(v_ns, device=device)[:min(batch_size, v_ns)]
                        if len(v_idx) >= 2:
                            v_yb = val_dataset.labels[v_idx]; v_wb = val_dataset.sample_weights[v_idx] if hasattr(val_dataset, 'sample_weights') and val_dataset.sample_weights is not None else None
                            vii, vjj = torch.randperm(len(v_idx), device=device), torch.randperm(len(v_idx), device=device)
                            v_si, v_sj = self.forward({k: v[v_idx][vii] for k, v in val_dataset.data.items()}), self.forward({k: v[v_idx][vjj] for k, v in val_dataset.data.items()})
                            v_target = torch.sign(v_yb[vii] - v_yb[vjj]).unsqueeze(1); v_mag = torch.abs(v_yb[vii] - v_yb[vjj]).unsqueeze(1); v_mag = v_mag / (v_mag.mean() + 1e-6)
                            v_loss_pw = criterion_pairwise(v_si, v_sj, v_target, weights=((v_wb[vii] + v_wb[vjj]).unsqueeze(1)/2.0 if v_wb is not None else 1.0) * v_mag).item()
                            v_loss_pt = (criterion_pointwise(v_si, v_yb[vii].unsqueeze(1)) + criterion_pointwise(v_sj, v_yb[vjj].unsqueeze(1))).item()
                            v_loss_val = v_loss_pw + (pointwise_lambda * v_loss_pt)
                    self.train(); scheduler.step(v_loss_val)
                if v_loss_val is not None:
                    if verbose: logger.info(f"   [Epoch {epoch+1}/{epochs}] Loss: {al:.6f} | Val Loss: {v_loss_val:.4f}")
                    if v_loss_val < best_val_loss: best_val_loss, wait = v_loss_val, 0; best_state = copy.deepcopy(self.state_dict())
                    else:
                        wait += 1
                        if wait >= patience: logger.warning(f"Early stopping at epoch {epoch+1}. Restoring best state (Val Loss: {best_val_loss:.4f})"); self.load_state_dict(best_state); break
                else: scheduler.step(al); logger.info(f"   [Epoch {epoch+1}/{epochs}] Loss: {al:.6f}")
            return
        if isinstance(dataset, tuple):
            X_all, y_all = dataset; ns = len(y_all); abs_ = min(batch_size, ns); nb = ns // abs_; champion_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            for epoch in range(epochs):
                tl, indices = 0, torch.randperm(ns, device=device)
                for b in range(nb):
                    optimizer.zero_grad(); idx = indices[b*abs_ : (b+1)*abs_]
                    if len(idx) < 2: continue
                    X, y = X_all[idx].to(device), y_all[idx].to(device); ii, jj = torch.randperm(len(idx), device=device), torch.randperm(len(idx), device=device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        si, sj = self.forward(X[ii]), self.forward(X[jj]); target = torch.sign(y[ii] - y[jj]).unsqueeze(1)
                        loss = criterion_pairwise(si, sj, target) + pointwise_lambda * (criterion_pointwise(si, y[ii].unsqueeze(1)) + criterion_pointwise(sj, y[jj].unsqueeze(1)))
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); tl += loss.item()
                champion_scheduler.step(); al = tl / max(1, nb); logger.info(f"   [Epoch {epoch+1}/{epochs}] Loss: {al:.6f}")
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
        else: torch.jit.trace(self, (torch.randn(1, self.model[0].in_features).to(device),), check_trace=False).save(path)

class MultiModalRankNet(RankNet):
    def __init__(self, specs: List[InputSpec] = None, hidden_dim: int = 64, **kwargs):
        if specs is None:
            scales, lookback = kwargs.get('scales', 32), kwargs.get('lookback', 63)
            specs = [InputSpec(name='x_seq', shape=(lookback, 1), type='seq'), InputSpec(name='x_spatial', shape=(1, scales, lookback), type='spatial'), InputSpec(name='x_momentum', shape=(lookback, 3), type='seq')]
        dropout = kwargs.get('dropout', 0.4)
        super(MultiModalRankNet, self).__init__(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.specs, self.hidden_dim, self.encoders, self.norms = specs, hidden_dim, nn.ModuleDict(), nn.ModuleDict()
        vh, gl = kwargs.get('vit_heads', 4), kwargs.get('gnn_layers', 2)
        # SENIOR FIX (V5.8.8): Increased complexity of Modality Gate and added learnable scaling
        self.gate = nn.Sequential(nn.Linear(hidden_dim * len(specs), hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(specs)), nn.Softmax(dim=1))
        self.modality_scales = nn.Parameter(torch.ones(len(specs)))
        self.spatial_dropout = nn.Dropout(p=0.3)
        for spec in specs:
            self.norms[spec.name] = nn.LayerNorm(hidden_dim)
            if spec.type == 'seq': self.encoders[spec.name] = nn.LSTM(input_size=spec.shape[-1], hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
            elif spec.type == 'spatial': self.encoders[spec.name] = LightweightViT(img_size=spec.shape[1:], patch_size=(spec.shape[1], 9), in_chans=spec.shape[0], embed_dim=hidden_dim, num_heads=vh, dropout=dropout)
            elif spec.type == 'graph': self.encoders[spec.name] = LightweightGNN(in_features=spec.shape[0], embed_dim=hidden_dim, depth=gl)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = []
        for i, spec in enumerate(self.specs):
            x = inputs[spec.name]
            if spec.type == 'seq': _, (h_n, _) = self.encoders[spec.name](x); e = h_n[-1]
            elif spec.type == 'spatial':
                e = self.encoders[spec.name](x)
                if self.training: e = self.spatial_dropout(e)
            else: e = self.encoders[spec.name](x)
            # Apply individual learnable scale and norm
            embs.append(self.norms[spec.name](e) * self.modality_scales[i])
        stacked_embs = torch.stack(embs, dim=1); flat_embs = torch.cat(embs, dim=1)
        weights = self.gate(flat_embs).unsqueeze(-1)
        return self.model((stacked_embs * weights).sum(dim=1))

class PairwiseRankLoss(nn.Module):
    def __init__(self, sigma: float = 1.0): super(PairwiseRankLoss, self).__init__(); self.sigma = sigma
    def forward(self, s_i, s_j, target, weights=None):
        loss = -((1 + target) / 2) * self.sigma * (s_i - s_j) + torch.log(1 + torch.exp(self.sigma * (s_i - s_j)))
        if weights is not None: loss = loss * weights
        return loss.mean()
