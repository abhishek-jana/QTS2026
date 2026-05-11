import torch
import numpy as np
import pandas as pd
import yaml
import os
import sys
from datetime import datetime, timedelta
from qts_core.logger import logger
from research_lab.alpha_universe import AlphaUniverse
from research_lab.data_engine import DataEngine
from research_lab.alpha_ranker import SkewAwareTransformer, InputSpec

def verify_ghost_signals():
    logger.info("🕵️ GHOST PROTOCOL: Component Verification Protocol")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # --- STEP 1: DATA INTEGRITY & SKEW CHECK ---
    logger.info("--- [Step 1: Data Audit] ---")
    engine = DataEngine(storage_path=config['data_engine']['storage_path'], read_only=True)
    universe = AlphaUniverse(conn=engine.conn, config=config)
    
    test_date = datetime(2024, 1, 15, 16, 0) # Peak AI Rally
    batch = universe.snapshot(as_of=test_date, tickers=config['universe']['tickers'])
    
    if batch is None:
        logger.error("❌ Data Extraction FAILED: Snapshot returned None.")
        return
        
    labels = batch.labels.numpy()
    logger.info(f"✅ Data Extraction SUCCESS: {len(batch.tickers)} stocks processed.")
    logger.info(f"📊 Label Distribution: Mean={np.mean(labels):.4f}, Std={np.std(labels):.4f}")
    
    # Check for Skew (Ghost Protocol Requirement)
    skew = pd.Series(labels).skew()
    logger.info(f"📈 Label Skew: {skew:.4f} (Target > 0 for Growth Hunter)")
    
    if np.std(labels) < 1e-6:
        logger.error("❌ SIGNAL COLLAPSE: Labels have zero variance.")
        return

    # --- STEP 2: MODEL ARCHITECTURE & DYNAMICS ---
    logger.info("--- [Step 2: Architecture Audit] ---")
    lookback = config['signal_physics']['lookback_days']
    n_scales = len(config['signal_physics']['wavelet_transform']['scales'])
    
    specs = [
        InputSpec(name='x_seq', shape=(lookback, 1), type='seq'),
        InputSpec(name='x_spatial', shape=(1, n_scales, lookback), type='spatial'),
        InputSpec(name='x_graph', shape=(8,), type='graph'),
        InputSpec(name='x_volume', shape=(lookback, 1), type='seq'),
        InputSpec(name='x_momentum', shape=(lookback, 3), type='seq')
    ]
    
    model = SkewAwareTransformer(specs=specs, embed_dim=128)
    logger.info("✅ SkewAwareTransformer Initialized.")
    
    # Test Forward Pass
    try:
        inputs = {k: v[:2] for k, v in batch.data.items()} # Take small batch
        out = model(inputs)
        std = out.std().item()
        logger.info(f"🧠 Model Dynamics: Score Std (Untrained) = {std:.6f}")
        
        if np.isnan(std):
            logger.error("❌ WEIGHT EXPLOSION: Model produced NaN scores.")
            return
    except Exception as e:
        logger.error(f"❌ FORWARD PASS FAILED: {e}")
        return

    # --- STEP 3: OPTIMIZER STABILITY ---
    logger.info("--- [Step 3: Optimization Audit] ---")
    try:
        # Mini-fit on 10 samples to check gradient flow
        mini_batch = {k: v[:16] for k, v in batch.data.items()}
        mini_labels = batch.labels[:16]
        
        from research_lab.alpha_universe import MultiModalBatch
        mini_dataset = MultiModalBatch(data=mini_batch, labels=mini_labels, tickers=['TEST']*16, times=[test_date]*16)
        
        logger.info("Testing gradient backprop...")
        model.fit(mini_dataset, epochs=2, batch_size=8, lr=0.001)
        logger.info("✅ Optimizer Stable: Gradient backprop completed without crash.")
    except Exception as e:
        logger.error(f"❌ OPTIMIZER FAILURE: {e}")
        return

    logger.success("🏁 GHOST PROTOCOL VERIFIED. System is ready for Full Production Run.")

if __name__ == "__main__":
    verify_ghost_signals()
