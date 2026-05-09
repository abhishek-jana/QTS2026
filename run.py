import sys
import os
import argparse
import yaml
from datetime import datetime
from qts_core.logger import logger
from dotenv import load_dotenv

# SENIOR DEV PATTERN: Programmatic Path Discovery & Env Loading
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

# Explicitly load .env file
load_dotenv(os.path.join(ROOT_DIR, ".env"))

def main():
    parser = argparse.ArgumentParser(description="UQTS-2026 Unified Execution Muscle")
    parser.add_argument("--ingest", action="store_true", help="Run full historical data ingestion")
    parser.add_argument("--train", action="store_true", help="Run 5-year model training")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, run evaluation only")
    parser.add_argument("--precompute-rl", action="store_true", help="Pre-compute data for Phase 3 RL training")

    subparsers = parser.add_subparsers(dest="command", help="System components")

    # Position-based commands
    subparsers.add_parser("lab", help="Run the Research Lab Orchestrator")
    subparsers.add_parser("prod", help="Run the Production Inference Worker")
    subparsers.add_parser("live", help="Run the Live Paper Trading Worker")
    subparsers.add_parser("ui", help="Run the Cockpit Backend server")
    subparsers.add_parser("sim", help="Run the High-Performance Simulation")

    # RL Command with its own arguments
    rl_parser = subparsers.add_parser("rl-train", help="Train the Phase 3 RL Portfolio Pilot")
    rl_parser.add_argument("--total-timesteps", type=int, default=100000, help="Number of training steps")

    args = parser.parse_args()

    # 0. Handle RL Pre-computation
    if args.precompute_rl:
        from scripts.precompute_rl_data import precompute_rl_data
        precompute_rl_data()
        return

    # 1. Handle Simulation & RL Training
    if args.command == "sim":
        from alpha_factory.simulation_engine import SimulationEngineV5
        sim = SimulationEngineV5()
        sim.run(datetime(2024, 1, 1), datetime(2026, 5, 1))
        return
    elif args.command == "rl-train":
        from scripts.train_rl_pilot import train_rl_pilot
        train_rl_pilot(total_timesteps=args.total_timesteps)
        return

    # 2. Handle Ingestion / Training (Lab Logic)
    if args.ingest or args.train or args.eval_only or args.command == "lab":
        from research_lab.backtest_comparison import BacktestOrchestrator
        orchestrator = BacktestOrchestrator()
        
        if args.ingest:
            orchestrator.run_ingestion()
        
        if args.train or (args.command == "lab" and not args.eval_only) or args.eval_only:
            # PRO EXPERT SETTING: Load timeframes from config
            tf = orchestrator.config['model_pipeline']['timeframes']
            
            def parse_date(d_str):
                if d_str == 'now': return datetime.now()
                return datetime.strptime(d_str, '%Y-%m-%d')

            orchestrator.run_comparison(
                parse_date(tf['train_start']), parse_date(tf['train_end']), 
                parse_date(tf['test_start']), parse_date(tf['test_end']),
                skip_train=args.eval_only
            )
            
    # 3. Handle Long-Running Workers
    if args.command in ["prod", "live"]:
        import asyncio
        from execution_muscle.inference_worker import InferenceWorker
        
        # Override config trading_mode if explicit 'live' command is used
        if args.command == "live":
            with open("config.yaml", "r") as f:
                conf = yaml.safe_load(f)
            conf['execution_muscle']['trading_mode'] = 'paper'
            with open("config.yaml", "w") as f:
                yaml.dump(conf, f, default_flow_style=False)
        
        worker = InferenceWorker()
        worker.initialize()
        asyncio.run(worker.run())
        
    elif args.command == "ui":
        import uvicorn
        uvicorn.run("cockpit_backend.main:app", host="0.0.0.0", port=8000, reload=True)
    
    elif not any([args.ingest, args.train, args.eval_only, args.command, args.precompute_rl]):
        parser.print_help()

if __name__ == "__main__":
    main()
