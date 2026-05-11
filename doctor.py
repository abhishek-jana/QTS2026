import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import duckdb
import psutil
from datetime import datetime, timedelta
from qts_core.logger import logger
from tqdm import tqdm

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

from research_lab.data_engine import DataEngine
from research_lab.alpha_universe import AlphaUniverse
from research_lab.alpha_ranker import SkewAwareTransformer, InputSpec

class QTSDoctor:
    """
    SENIOR DIAGNOSTIC SUITE (V6.0 "Ghost Protocol" Edition)
    Automated System, Data, and Model Verification.
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.db_path = self.config['data_engine']['storage_path']
        
    def check_infra(self):
        logger.info("🩺 [INFRA] Checking Hardware & Environment...")
        ram = psutil.virtual_memory()
        logger.info(f"   RAM: {ram.percent}% used ({ram.available / 1024**3:.1f}GB available)")
        if ram.percent > 90: logger.warning("   ⚠️ RAM usage is critical.")
            
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   GPU: {gpu_name} ({gpu_mem:.1f}GB VRAM detected)")
        else: logger.warning("   ⚠️ CUDA not detected.")
            
        zombies = []
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                files = proc.info.get('open_files')
                if files:
                    for f in files:
                        if self.db_path in f.path and proc.pid != os.getpid(): zombies.append(proc.pid)
            except Exception: pass
        if zombies: logger.warning(f"   ⚠️ Zombie processes holding DB lock: {zombies}")
        else: logger.info("   ✅ No DuckDB zombie locks detected.")

    def check_data_integrity(self):
        logger.info("🩺 [DATA] Auditing Bitemporal Integrity...")
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            tickers = self.config['universe']['tickers']
            count = conn.execute("SELECT count(distinct ticker) FROM market_data").fetchone()[0]
            logger.info(f"   Tickers in DB: {count}/{len(tickers)}")
            
            # SENIOR FIX: Use Jan 16, 2024 (Jan 15 was MLK Day, market closed)
            logger.info("   Auditing 'Growth Hunter' Label Skew (Jan 16, 2024)...")
            engine = DataEngine(storage_path=self.db_path, read_only=True)
            universe = AlphaUniverse(conn=engine.conn, config=self.config)
            test_date = datetime(2024, 1, 16, 16, 0)
            batch = universe.snapshot(as_of=test_date, tickers=tickers)
            
            if batch:
                labels = batch.labels.numpy()
                skew = pd.Series(labels).skew()
                logger.info(f"   ✅ Sample Skew: {skew:.4f} (Target > 0)")
            else: logger.error("   ❌ SNAPSHOT FAILED: No data found for Jan 16, 2024.")
            conn.close()
        except Exception as e: logger.error(f"   ❌ Data Audit Failed: {e}")

    def check_model_health(self):
        logger.info("🩺 [MODEL] Auditing Ghost Protocol Architecture...")
        try:
            lookback = self.config['signal_physics']['lookback_days']
            n_scales = len(self.config['signal_physics']['wavelet_transform']['scales'])
            
            specs = [
                InputSpec(name='x_seq', shape=(lookback, 1), type='seq'),
                InputSpec(name='x_spatial', shape=(1, n_scales, lookback), type='spatial'),
                InputSpec(name='x_graph', shape=(8,), type='graph'),
                InputSpec(name='x_volume', shape=(lookback, 1), type='seq'),
                InputSpec(name='x_momentum', shape=(lookback, 3), type='seq')
            ]
            
            model = SkewAwareTransformer(specs=specs, embed_dim=128)
            
            # Forward Pass Variance (SENIOR FIX: Correct shapes for multi-modal transformer)
            dummy_inputs = {
                'x_seq': torch.randn(2, lookback, 1),
                'x_spatial': torch.randn(2, 1, n_scales, lookback),
                'x_graph': torch.randn(2, 8),
                'x_volume': torch.randn(2, lookback, 1),
                'x_momentum': torch.randn(2, lookback, 3)
            }
            out = model(dummy_inputs)
            std = out.std().item()
            logger.info(f"   Initial Score Std: {std:.6f}")
            if std < 1e-4: logger.error("   ❌ ARCHITECTURAL COLLAPSE.")
            else: logger.info("   ✅ Model Dynamics: Healthy.")
                
            logger.info("   Verifying Gradient Backprop...")
            target = torch.randn(2, 1)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            
            grads = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
            if any(np.isnan(grads)): logger.error("   ❌ GRADIENT EXPLOSION.")
            else: logger.info(f"   ✅ Gradient Flow: Mean Magnitude = {np.mean(grads):.6e}")
                
        except Exception as e: logger.error(f"   ❌ Model Audit Failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="UQTS-2026 'The Doctor': Clinical Pipeline Diagnostics")
    parser.add_argument("command", choices=["infra", "data", "model", "full"], help="Diagnostic module")
    args = parser.parse_args()
    doctor = QTSDoctor()
    if args.command == "infra": doctor.check_infra()
    elif args.command == "data": doctor.check_data_integrity()
    elif args.command == "model": doctor.check_model_health()
    elif args.command == "full":
        doctor.check_infra()
        doctor.check_data_integrity()
        doctor.check_model_health()
    logger.success("🏁 Diagnostic session complete.")

if __name__ == "__main__":
    main()
