import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from qts_core.logger import logger
from research_lab.alpha_universe import MultiModalBatch

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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
        # x is [batch, 1, scales, lookback]
        p = self.patcher(x); b, c, h, w = p.shape
        tokens = p.view(b, c, -1).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(b, -1, -1), tokens), dim=1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out); x = self.norm2(x + self.ff(x))
        return x[:, 0] # Return CLS token embedding

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, context_dim: int = None):
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = GatedLinearUnit(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x, context=None):
        residual = self.skip_proj(x)
        x = self.fc1(x)
        if context is not None and self.context_fc is not None:
            x = x + self.context_fc(context)
        x = F.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        return self.norm(x + residual)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int, dropout: float = 0.1, context_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dims = input_dims
        
        self.grns = nn.ModuleList([
            GatedResidualNetwork(dim, hidden_dim, hidden_dim, dropout) for dim in input_dims
        ])
        
        combined_input_dim = len(input_dims) * hidden_dim
        self.selection_grn = GatedResidualNetwork(combined_input_dim, hidden_dim, len(input_dims), dropout, context_dim)

    def forward(self, x: List[torch.Tensor], context=None):
        var_outputs = [self.grns[i](x[i]) for i in range(len(x))]
        flattened = torch.cat(var_outputs, dim=-1)
        weights = self.selection_grn(flattened, context)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)
        stacked_outputs = torch.stack(var_outputs, dim=-2)
        combined = (stacked_outputs * weights).sum(dim=-2)
        return combined, weights.squeeze(-1)

class QuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - pred[:, i:i+1]
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss.mean())
        return sum(losses)

class SniperTFT(nn.Module):
    """
    Temporal Fusion Transformer tailored for the Sniper-Residual Strategy.
    Optimized for a small 60-stock universe.
    """
    def __init__(self, 
                 past_input_dims: Dict[str, int], 
                 static_input_dims: Dict[str, int],
                 hidden_dim: int = 32, 
                 attn_heads: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.quantiles = [0.1, 0.5, 0.9]
        
        # Spatial Processing (ViT)
        # Using embed_dim matching hidden_dim for clean integration
        self.spatial_vit = LightweightWaveletViT(in_chans=1, embed_dim=hidden_dim, patch_size=(4, 4))
        
        # Sort keys to ensure deterministic order matching forward pass
        self.static_keys = sorted(static_input_dims.keys())
        self.past_keys = sorted([k for k in past_input_dims.keys() if k != 'x_spatial'])
        
        # Static variable selection
        self.static_vsn = VariableSelectionNetwork(
            input_dims=[static_input_dims[k] for k in self.static_keys],
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # We concatenate the ViT embedding with the static VSN embedding
        # so the static context is [2 * hidden_dim] before projection
        self.c_selection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_enrichment = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_state = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Past variable selection (Temporal)
        self.past_vsn = VariableSelectionNetwork(
            input_dims=[past_input_dims[k] for k in self.past_keys],
            hidden_dim=hidden_dim,
            dropout=dropout,
            context_dim=hidden_dim
        )
        
        # Temporal Processing (LSTM)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.post_lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)
        
        # Static Enrichment
        self.static_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout, context_dim=hidden_dim)
        
        # Temporal Self-Attention
        self.attn = nn.MultiheadAttention(hidden_dim, attn_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Final Output Layer (Quantiles)
        self.output_layer = nn.Linear(hidden_dim, len(self.quantiles))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Process Spatial ViT
        spatial_input = inputs['past']['x_spatial'] # [batch, 1, scales, seq]
        vit_emb = self.spatial_vit(spatial_input) # [batch, hidden_dim]
        
        # Process Static
        static_vars = [inputs['static'][name] for name in self.static_keys]
        static_emb, static_weights = self.static_vsn(static_vars) # [batch, hidden_dim]
        
        # Combine ViT and Static to form rich global context
        global_context = torch.cat([static_emb, vit_emb], dim=-1) # [batch, 2*hidden_dim]
        
        # Context vectors
        cs = self.c_selection(global_context)
        ce = self.c_enrichment(global_context)
        ch = self.c_state(global_context)
        
        # Past VSN
        past_vars = [inputs['past'][name] for name in self.past_keys]
        cs_expanded = cs.unsqueeze(1) # [batch, 1, hidden_dim]
        past_emb, past_weights = self.past_vsn(past_vars, cs_expanded) # [batch, seq_len, hidden_dim]
        
        # LSTM Processing
        h0 = ch.unsqueeze(0) # [1, batch, hidden_dim]
        c0 = torch.zeros_like(h0)
        lstm_out, _ = self.lstm(past_emb, (h0, c0)) # [batch, seq_len, hidden_dim]
        
        # Skip connection from past_emb
        lstm_out = self.post_lstm_norm(self.post_lstm_gate(lstm_out) + past_emb)
        
        # Static Enrichment
        ce_expanded = ce.unsqueeze(1)
        enriched = self.static_enrichment(lstm_out, ce_expanded) # [batch, seq_len, hidden_dim]
        
        # Temporal Self-Attention
        attn_out, _ = self.attn(enriched, enriched, enriched)
        x = self.attn_norm(attn_out + enriched)
        
        # Take the last step for prediction
        last_step = x[:, -1, :] # [batch, hidden_dim]
        
        out = self.output_layer(last_step) # [batch, 3]
        
        return {
            'out': out,
            'static_weights': static_weights,
            'past_weights': past_weights, 
        }

class SniperRanker(nn.Module):
    def __init__(self, specs: Dict[str, Dict[str, int]], hidden_dim: int = 32):
        super().__init__()
        self.model = SniperTFT(
            past_input_dims=specs['past'],
            static_input_dims=specs['static'],
            hidden_dim=hidden_dim
        )
        self.criterion = QuantileLoss()

    def forward(self, batch: MultiModalBatch) -> torch.Tensor:
        tft_inputs = {
            'static': {k.replace('x_static_', ''): v for k, v in batch.data.items() if k.startswith('x_static_')},
            'past': {k.replace('x_past_', ''): v for k, v in batch.data.items() if k.startswith('x_past_')}
        }
        return self.model(tft_inputs)['out']

    @torch.jit.ignore
    def fit(self, dataset: MultiModalBatch, epochs: int = 50, lr: float = 0.001, 
            device: torch.device = torch.device('cpu'), batch_size: int = 128,
            val_dataset: MultiModalBatch = None, patience: int = 15):
        import copy
        self.to(device)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # SENIOR FIX (Convergence): Use a scheduler to cool down the LR on plateaus.
        # This helps the model "squeeze" the final alpha out of the noise.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # EFFICIENCY (AMP): Use Automatic Mixed Precision for 2-3x faster throughput
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        
        ns = len(dataset.labels)
        nb = max(1, ns // batch_size)

        logger.info(f"SniperRanker: Training on {ns} samples over {epochs} epochs...")

        # EFFICIENCY: move training and validation data to the target device ONCE
        # before the epoch loop. The original code did v[idx].to(device) inside
        # every mini-batch AND val_dataset.to(device) every epoch:
        #   - (ns / batch_size) * epochs CPU->GPU transfers for training data
        #   - epochs CPU->GPU transfers for the val set
        # With a ~78K sample training set and batch_size=512, that is up to
        # ~15,000 redundant transfers per run. Moving once upfront eliminates
        # all of them. Falls back to per-batch transfer on OOM.
        train_on_device = None
        labels_on_device = None
        if device.type != 'cpu':
            try:
                train_on_device  = {k: v.to(device, non_blocking=True) for k, v in dataset.data.items()}
                labels_on_device = dataset.labels.to(device, non_blocking=True)
                logger.info(f"SniperRanker: Training data pinned to {device}.")
            except RuntimeError as e:
                logger.warning(f"SniperRanker: Cannot pin training data to {device} ({e}). "
                               "Falling back to per-batch transfer.")
                train_on_device  = None
                labels_on_device = None

        val_on_device = None
        if val_dataset is not None and device.type != 'cpu':
            try:
                val_on_device = val_dataset.to(device)
            except RuntimeError:
                val_on_device = None

        best_val_ic = -float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.train()
            trl = 0
            perm = torch.randperm(ns)

            for b in range(nb):
                optimizer.zero_grad(set_to_none=True)
                idx = perm[b * batch_size:(b + 1) * batch_size]
                if len(idx) < 2:
                    continue

                if train_on_device is not None:
                    batch_data   = {k: v[idx] for k, v in train_on_device.items()}
                    batch_labels = labels_on_device[idx].unsqueeze(1)
                else:
                    batch_data   = {k: v[idx].to(device) for k, v in dataset.data.items()}
                    batch_labels = dataset.labels[idx].to(device).unsqueeze(1)

                batch_obj = MultiModalBatch(data=batch_data, labels=batch_labels, tickers=[], times=[])
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        preds = self.forward(batch_obj)
                        loss  = self.criterion(preds, batch_labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = self.forward(batch_obj)
                    loss  = self.criterion(preds, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()

                trl += loss.item()

            if val_dataset:
                self.eval()
                with torch.no_grad():
                    v_batch = val_on_device if val_on_device is not None else val_dataset.to(device)
                    # Run validation in AMP if available
                    if device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            v_preds = self.forward(v_batch)
                    else:
                        v_preds = self.forward(v_batch)
                        
                    v_loss  = self.criterion(v_preds, v_batch.labels.unsqueeze(1))
                    
                    from scipy.stats import spearmanr
                    ic = spearmanr(v_preds[:, 1].cpu().numpy(), val_dataset.labels.cpu().numpy())[0]
                
                # Update scheduler based on Val IC
                scheduler.step(ic)
                    
                logger.info(f"   [Epoch {epoch+1}] Loss: {trl/nb:.4f} | Val Loss: {v_loss:.4f} | Val IC: {ic:.4f}")
                
                if ic > best_val_ic:
                    best_val_ic = ic
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.state_dict())
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.warning(f"   [Early Stopping] Triggered at Epoch {epoch+1}. Best IC: {best_val_ic:.4f}")
                    if best_model_state:
                        self.load_state_dict(best_model_state)
                    break
            else:
                logger.info(f"   [Epoch {epoch+1}] Loss: {trl/nb:.4f}")

    @torch.jit.ignore
    def predict(self, batch: MultiModalBatch) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(batch)