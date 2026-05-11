import sys
import os
import argparse
import yaml
import json
import random
import numpy as np
import torch
from datetime import datetime
from qts_core.logger import logger
from dotenv import load_dotenv

# SENIOR DEV PATTERN: Programmatic Path Discovery & Env Loading
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

# Explicitly load .env file
load_dotenv(os.path.join(ROOT_DIR, ".env"))

def set_seed(seed=42):
    """Ensures bit-level reproducibility across all components."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.debug(f"Reproducibility Mode: Random Seed {seed} Locked.")

def cleanup_zombie_locks():
    """SENIOR AUTOMATION: Proactively release DuckDB locks. Non-breaking if psutil missing."""
    try:
        import psutil
        import signal
    except ImportError:
        logger.warning("⚠️ 'psutil' not found. Skipping zombie lock cleanup. System may hit DuckDB IO Errors.")
        return

    try:
        with open("config.yaml", "r") as f:
            conf = yaml.safe_load(f)
        db_path = conf.get('data_engine', {}).get('storage_path', 'data/uqts_v2_intraday.ddb')
        db_full_path = os.path.abspath(db_path)
        
        current_pid = os.getpid()
        
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                files = proc.info.get('open_files')
                if files:
                    for f in files:
                        if f.path == db_full_path and proc.info['pid'] != current_pid:
                            logger.warning(f"🧹 Auto-Cleanup: Evicting zombie process {proc.info['pid']} from DB lock.")
                            os.kill(proc.info['pid'], signal.SIGKILL)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.debug(f"Cleanup failed (non-critical): {e}")

def parse_date(d_str):
    if d_str == 'now': return datetime.now()
    return datetime.strptime(d_str, '%Y-%m-%d')

def main():
    parser = argparse.ArgumentParser(description="UQTS-2026 Unified Intelligence CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    subparsers = parser.add_subparsers(dest="command", help="System components")

    # Load config for smart defaults
    try:
        with open("config.yaml", "r") as f:
            conf = yaml.safe_load(f)
        default_steps = conf.get('model_pipeline', {}).get('rl_training', {}).get('total_timesteps', 1000000)
    except Exception:
        default_steps = 1000000

    # --- 1. Signal Pipeline (The Brain / Prediction Engine) ---
    signal_parser = subparsers.add_parser("signal", help="Supervised Signal Extraction (e.g. Ghost Protocol Transformer)")
    signal_sub = signal_parser.add_subparsers(dest="subcommand")
    signal_sub.add_parser("ingest", help="Ingest historical market data")
    signal_sub.add_parser("train", help="Train the signal extraction model")
    signal_sub.add_parser("eval", help="Evaluate signal accuracy (IC/WinRate)")

    # --- 2. RL Pipeline (The General / Allocation Engine) ---
    rl_parser = subparsers.add_parser("rl", help="Reinforcement Learning Allocation")
    rl_sub = rl_parser.add_subparsers(dest="subcommand")
    rl_sub.add_parser("data", help="Pre-compute training data for RL")
    train_parser = rl_sub.add_parser("train")
    train_parser.add_argument("--steps", type=int, default=default_steps)
    rl_sub.add_parser("eval", help="Unified portfolio evaluation")

    # --- 3. Full Pipeline ---
    full_parser = subparsers.add_parser("full", help="Execute complete Signal -> RL pipeline")
    full_parser.add_argument("--steps", type=int, default=default_steps)

    # --- 4. Operations ---
    subparsers.add_parser("prod", help="Launch Production Inference Worker")
    subparsers.add_parser("live", help="Launch Live Paper Trading Worker")
    subparsers.add_parser("ui", help="Launch Mission Control Cockpit")

    args = parser.parse_args()

    # SENIOR DEV LOGIC: Always be reproducible, always be clean.
    set_seed(args.seed)
    if args.command in ["full", "signal", "rl"]:
        cleanup_zombie_locks()

    # --- ROUTING ---

    if args.command == "full":
        logger.info("🚀 EXECUTING FULL PRODUCTION SUITE (V6.0 GHOST PROTOCOL)")
        from research_lab.backtest_comparison import BacktestOrchestrator
        orch = BacktestOrchestrator()
        tf = orch.config['model_pipeline']['timeframes']
        
        logger.info("--- PHASE 1: SIGNAL EXTRACTION ---")
        orch.run_comparison(parse_date(tf['train_start']), parse_date(tf['train_end']),
                           parse_date(tf['test_start']), parse_date(tf['test_end']))
        
        logger.info("--- PHASE 2: SENSOR DATA PRE-COMPUTATION ---")
        from scripts.precompute_rl_data import precompute_rl_data
        precompute_rl_data()
        
        logger.info("--- PHASE 3: ALLOCATION POLICY TRAINING ---")
        from scripts.train_rl_pilot import train_rl_pilot
        train_rl_pilot(total_timesteps=args.steps)
        
        logger.info("--- PHASE 4: UNIFIED SYSTEM EVALUATION ---")
        from scripts.rl_evaluator import run_rl_evaluation
        run_rl_evaluation()
        
        logger.success("🏁 GHOST PROTOCOL PIPELINE COMPLETED.")

    elif args.command == "signal":
        from research_lab.backtest_comparison import BacktestOrchestrator
        orch = BacktestOrchestrator()
        if args.subcommand == "ingest": orch.run_ingestion()
        elif args.subcommand == "train":
            tf = orch.config['model_pipeline']['timeframes']
            orch.run_comparison(parse_date(tf['train_start']), parse_date(tf['train_end']),
                               parse_date(tf['test_start']), parse_date(tf['test_end']), skip_train=False)
        elif args.subcommand == "eval":
            tf = orch.config['model_pipeline']['timeframes']
            orch.run_comparison(parse_date(tf['train_start']), parse_date(tf['train_end']),
                               parse_date(tf['test_start']), parse_date(tf['test_end']), skip_train=True)
        elif args.subcommand == "test-subset":
            from research_lab.backtest_comparison import BacktestOrchestrator
            orch = BacktestOrchestrator(tickers=["SPY", "NVDA", "TSM"])
            tf = orch.config['model_pipeline']['timeframes']
            import datetime as dt
            clean_train_end = parse_date(tf['train_end']) - dt.timedelta(days=5)
            orch.run_comparison(parse_date(tf['train_start']), clean_train_end,
                               parse_date(tf['test_start']), parse_date(tf['test_end']), skip_train=False)

    elif args.command == "rl":
        if args.subcommand == "data":
            from scripts.precompute_rl_data import precompute_rl_data
            precompute_rl_data()
        elif args.subcommand == "train":
            from scripts.train_rl_pilot import train_rl_pilot
            train_rl_pilot(total_timesteps=args.steps)
        elif args.subcommand == "eval":
            from scripts.rl_evaluator import run_rl_evaluation
            run_rl_evaluation()

    elif args.command in ["prod", "live"]:
        import asyncio
        from execution_muscle.inference_worker import InferenceWorker
        if args.command == "live":
            with open("config.yaml", "r") as f: conf = yaml.safe_load(f)
            conf['execution_muscle']['trading_mode'] = 'paper'
            with open("config.yaml", "w") as f: yaml.dump(conf, f)
        worker = InferenceWorker(); worker.initialize(); asyncio.run(worker.run())
        
    elif args.command == "ui":
        import uvicorn
        uvicorn.run("cockpit_backend.main:app", host="0.0.0.0", port=8000, reload=True)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
