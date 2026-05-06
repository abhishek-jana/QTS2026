import sys
import os
import argparse
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
    subparsers = parser.add_subparsers(dest="command", help="System components")

    lab_parser = subparsers.add_parser("lab", help="Run the Research Lab")
    lab_parser.add_argument("--ingest", action="store_true", help="Run data ingestion")
    lab_parser.add_argument("--train", action="store_true", help="Run model training")
    lab_parser.add_argument("--eval-only", action="store_true", help="Evaluation only")
    lab_parser.add_argument("--test-subset", action="store_true", help="Quick test subset")

    subparsers.add_parser("prod", help="Run the Production Inference Worker (Sim Mode)")
    subparsers.add_parser("live", help="Run the Live Paper Trading Worker (Alpaca Mode)")
    subparsers.add_parser("ui", help="Run the Cockpit Backend server")

    args = parser.parse_args()

    if args.command == "lab":
        from research_lab.backtest_comparison import BacktestOrchestrator
        orchestrator = BacktestOrchestrator(tickers=["SPY", "NVDA", "TSM"] if args.test_subset else None)
        if args.ingest: orchestrator.run_ingestion()
        
        do_train = args.train or (not args.ingest and not args.eval_only)
        if do_train or args.eval_only:
            orchestrator.run_comparison(
                datetime(2018, 1, 1), datetime(2022, 12, 31), 
                datetime(2023, 1, 1), datetime.now(),
                skip_train=args.eval_only
            )
    elif args.command in ["prod", "live"]:
        import asyncio
        import yaml
        from execution_muscle.inference_worker import InferenceWorker
        
        # Override config trading_mode if explicit 'live' command is used
        if args.command == "live":
            with open("config.yaml", "r") as f:
                conf = yaml.safe_load(f)
            conf['execution_muscle']['trading_mode'] = 'paper'
            with open("config.yaml", "w") as f:
                yaml.dump(conf, f)
        
        worker = InferenceWorker()
        worker.initialize()
        asyncio.run(worker.run())
    elif args.command == "ui":
        import uvicorn
        uvicorn.run("cockpit_backend.main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
