import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from qts_core.logger import logger
from scipy.stats import spearmanr

@dataclass
class InputSpec:
    name: str
    shape: Tuple[int, ...]
    type: str

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x): return self._norm(x.float()).type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.w1 = nn.Linear(dim, dim * 2); self.w2 = nn.Linear(dim, dim)
    def forward(self, x):
        x, gate = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * x)

class LightweightWaveletViT(nn.Module):
    def __init__(self, in_chans=1, embed_dim=128, patch_size=(4, 4)):
        super().__init__()
        self.patcher = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim); self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.apply(init_weights)
    def forward(self, x):
        p = self.patcher(x); b, c, h, w = p.shape
        tokens = p.view(b, c, -1).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(b, -1, -1), tokens), dim=1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out); x = self.norm2(x + self.ff(x))
        return x[:, 0]

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, embed_dim // 2, num_layers=layers, batch_first=True, bidirectional=True)
        self.norm = RMSNorm(embed_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = torch.cat((h[-2], h[-1]), dim=-1)
        return self.norm(out)

class WeightedSumMixer(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs) / n_inputs)
    def forward(self, inputs: List[torch.Tensor]):
        w = F.softmax(self.weights, dim=0)
        return sum(w[i] * inputs[i] for i in range(len(inputs)))

class PowerScaledAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, skew_init: float = 2.0):
        super().__init__()
        self.heads, self.dim_head = heads, dim // heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False); self.to_out = nn.Linear(dim, dim)
        self.skew = nn.Parameter(torch.full((heads,), skew_init) + torch.randn(heads) * 0.01) 
    def forward(self, x):
        b, n, _ = x.shape; h = self.heads; qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, self.dim_head).transpose(1, 2), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        skew = self.skew.view(1, h, 1, 1).clamp(0.1, 4.0) 
        attn = (dots * skew).softmax(dim=-1); out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class AsymmetricMagnitudeLoss(nn.Module):
    def __init__(self, miss_penalty: float = 2.0, false_alarm_penalty: float = 1.0):
        super().__init__()
        self.miss_penalty, self.fa_penalty = miss_penalty, false_alarm_penalty
    def forward(self, pred, target):
        err = target - pred; weight = torch.where(err > 0, self.miss_penalty, self.fa_penalty)
        return (weight * (err ** 2)).mean()

class SkewAwareTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, skip_norm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(dim) if not skip_norm else nn.Identity()
        self.attn = PowerScaledAttention(dim, heads=heads)
        self.norm2 = RMSNorm(dim); self.ff = SwiGLU(dim)
        self.gamma1 = nn.Parameter(1e-4 * torch.ones(dim)); self.gamma2 = nn.Parameter(1e-4 * torch.ones(dim))
    def forward(self, x):
        x = x + self.gamma1 * self.attn(self.norm1(x)); x = x + self.gamma2 * self.ff(self.norm2(x))
        return x

class SkewAwareTransformer(nn.Module):
    def __init__(self, specs: List[InputSpec], embed_dim: int = 128, depth: int = 4, heads: int = 8):
        super().__init__()
        self.specs = specs; self.embed_dim = embed_dim
        self.encoders = nn.ModuleDict(); self.mixer = None; n_temp = 0
        for spec in specs:
            if spec.name == 'raw_price': continue
            if spec.type == 'spatial': 
                self.encoders[spec.name] = LightweightWaveletViT(in_chans=1, embed_dim=embed_dim); n_temp += 1
            elif spec.type == 'seq':
                self.encoders[spec.name] = LSTMEncoder(input_dim=spec.shape[-1], embed_dim=embed_dim); n_temp += 1
            elif spec.type == 'graph':
                self.encoders[spec.name] = nn.Sequential(nn.Linear(spec.shape[0], embed_dim), RMSNorm(embed_dim))
        if n_temp > 1: self.mixer = WeightedSumMixer(n_temp)
        self.log_var_rank = nn.Parameter(torch.zeros(1)); self.log_var_mag = nn.Parameter(torch.zeros(1))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02) 
        blocks = [SkewAwareTransformerBlock(embed_dim, heads, skip_norm=True)]
        for _ in range(depth - 1): blocks.append(SkewAwareTransformerBlock(embed_dim, heads, skip_norm=False))
        self.transformer = nn.Sequential(*blocks)
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.SiLU(), nn.Linear(embed_dim // 2, 1), nn.Tanh())
        self.head_scale = 0.2
        self.apply(init_weights)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        b = next(iter(inputs.values())).shape[0]; temp_embs = []; static_embs = []
        for spec in self.specs:
            if spec.name not in self.encoders: continue
            emb = self.encoders[spec.name](inputs[spec.name])
            if spec.type in ['spatial', 'seq']: temp_embs.append(emb)
            else: static_embs.append(emb)
        unified_temp = self.mixer(temp_embs).unsqueeze(1) if self.mixer and temp_embs else (temp_embs[0].unsqueeze(1) if temp_embs else None)
        cls_token = self.cls_token.expand(b, -1, -1)
        if static_embs: cls_token = cls_token + torch.stack(static_embs).mean(dim=0).unsqueeze(1)
        x = torch.cat((cls_token, unified_temp), dim=1) if unified_temp is not None else cls_token
        x = x + self.pos_embed[:, :x.size(1), :]
        return self.head(self.norm(self.transformer(x)[:, 0])) * self.head_scale

    @torch.jit.ignore
    def predict_dataset(self, dataset, batch_size: int = 512) -> torch.Tensor:
        self.eval(); scores = []; ns = len(dataset.labels); device = next(self.parameters()).device
        for i in range(0, ns, batch_size):
            batch = {k: v[i:i+batch_size].to(device) for k, v in dataset.data.items()}
            scores.append(self.forward(batch))
        return torch.cat(scores)

    @torch.jit.ignore
    def fit(self, dataset, epochs: int = 50, lr: float = 0.0003, device: torch.device = torch.device('cpu'), 
            batch_size: int = 1024, patience: int = 30, weight_decay: float = 1e-4, val_dataset: Any = None):
        seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.to(device); self.train()
        p_base, p_juice = [], []
        for n, p in self.named_parameters():
            if any(k in n for k in ['skew', 'log_var']): p_juice.append(p)
            else: p_base.append(p)
        optimizer = optim.AdamW([{'params': p_base}, {'params': p_juice, 'lr': lr * 5.0}], lr=lr, weight_decay=weight_decay)
        total_steps = epochs * (max(1, len(dataset.labels) // batch_size) + 1)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*2, total_steps=total_steps)
        criterion_rank = PairwiseRankLoss(); criterion_mag = AsymmetricMagnitudeLoss()
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        best_val_ic, wait, best_state = -1.0, 0, None; ns = len(dataset.labels); nb = max(1, ns // batch_size)
        for epoch in range(epochs):
            trl_r, trl_m = 0, 0; perm = torch.randperm(ns, device=device)
            for b in range(nb):
                optimizer.zero_grad(set_to_none=True); idx = perm[b*batch_size:(b+1)*batch_size]
                if len(idx) < 2: continue
                yb = dataset.labels[idx]; yb_leaky = torch.where(yb > 0, yb, yb * 0.1) 
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    scores = self.forward({k: v[idx] for k, v in dataset.data.items()})
                    ii, jj = torch.randperm(len(idx), device=device), torch.randperm(len(idx), device=device)
                    si, sj = scores[ii], scores[jj]; yi, yj = yb_leaky[ii].unsqueeze(1), yb_leaky[jj].unsqueeze(1)
                    l_r = criterion_rank(si, sj, torch.sign(yi - yj)); l_m = criterion_mag(si, yi) + criterion_mag(sj, yj)
                    v_r = self.log_var_rank.clamp(-2, 2); v_m = self.log_var_mag.clamp(-2, 2)
                    loss = (l_r * torch.exp(-v_r) + v_r + l_m * torch.exp(-v_m) + v_m)
                if not torch.isnan(loss):
                    scaler.scale(loss).backward(); torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    scaler.step(optimizer); scaler.update(); trl_r += l_r.item(); trl_m += l_m.item()
                    scheduler.step()
            if val_dataset:
                self.eval(); 
                with torch.no_grad():
                    v_out = self.predict_dataset(val_dataset, batch_size=batch_size)
                    v_ic = spearmanr(v_out.squeeze().cpu().numpy(), val_dataset.labels.cpu().numpy())[0]
                self.train()
                logger.info(f"   [Epoch {epoch+1}] Val IC: {v_ic:.4f} | R-Loss: {trl_r/nb:.3f} | M-Loss: {trl_m/nb:.3f}")
                if epoch > (epochs * 0.2):
                    if v_ic > best_val_ic: best_val_ic, wait = v_ic, 0; best_state = copy.deepcopy(self.state_dict())
                    else:
                        wait += 1
                        if wait >= patience: logger.warning("Early Stop."); break
                elif v_ic > best_val_ic: best_val_ic = v_ic; best_state = copy.deepcopy(self.state_dict())
        if best_state: self.load_state_dict(best_state)

    @torch.jit.ignore
    def export(self, path: str):
        self.eval(); device = next(self.parameters()).device
        example_inputs = {spec.name: torch.randn(1, *spec.shape).to(device) for spec in self.specs if spec.name != 'raw_price'}
        torch.jit.trace(self, (example_inputs,), check_trace=False).save(path)

class PairwiseRankLoss(nn.Module):
    def __init__(self, sigma: float = 1.0): super().__init__(); self.sigma = sigma
    def forward(self, s_i, s_j, target):
        return F.softplus(-target * self.sigma * (s_i - s_j)).mean()

class RankNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, x): return self.model(x)
    @torch.jit.ignore
    def fit(self, dataset, epochs: int = 50, lr: float = 0.001, device: torch.device = torch.device('cpu'), batch_size: int = 1024, **kwargs):
        self.to(device); self.train(); optimizer = optim.Adam(self.parameters(), lr=lr)
        X, y = dataset; ns = len(y); nb = max(1, ns // batch_size)
        for epoch in range(epochs):
            tl = 0; perm = torch.randperm(ns, device=device)
            for b in range(nb):
                optimizer.zero_grad(set_to_none=True); idx = perm[b*batch_size:(b+1)*batch_size]
                if len(idx) < 2: continue
                loss = F.mse_loss(self.forward(X[idx]), y[idx].unsqueeze(1))
                loss.backward(); optimizer.step(); tl += loss.item()
            if epoch % 10 == 0: logger.info(f"   [MLP Epoch {epoch}] Loss: {tl/nb:.6f}")
