import asyncio
import yaml
from execution_muscle.oms import OMS
from cockpit_backend.streamer import DataStreamer

async def run_paper_bot():
    """
    UQTS-2026 Paper Trading Bot
    Orchestrates the Live Ingestion -> Inference -> OMS Rebalance loop.
    """
    print("🤖 UQTS-2026 Paper Trading Bot: INITIALIZING")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 1. Initialize OMS & Research Lab
    oms = OMS(config)
    
    from research_lab.alpha_universe import AlphaUniverse
    from research_lab.plugins.core_plugins import SequentialPlugin, SpatialPlugin
    from research_lab.real_data_ingestor import YFinanceIngestor
    from research_lab.alpha_ranker import MultiModalRankNet
    from datetime import datetime
    import torch
    
    lab = AlphaUniverse(plugins=[SequentialPlugin(), SpatialPlugin()])
    ingestor = YFinanceIngestor(lab.engine)
    
    # 2. Load the trained "Brain" (TorchScript Archive)
    try:
        model = torch.jit.load("model.pt")
        model.eval()
        print("✅ Trained TorchScript RankNet Loaded.")
    except Exception as e:
        print(f"⚠️ TorchScript load failed ({e}). Using random initialization (Python) for safety.")
        model = MultiModalRankNet(scales=32, lookback=63)
        model.eval()

    while True:
        try:
            # 1. Market Clock Check
            clock = oms.api.get_clock()
            if not clock.is_open:
                print(f"💤 Market is CLOSED. Next open: {clock.next_open.strftime('%Y-%m-%d %H:%M')}")
                await asyncio.sleep(900)
                continue

            now = datetime.now()
            print(f"\n🕒 Loop Start: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 2. Ingestion (Latest Market Reality)
            ingestor.ingest_universe(config['universe']['tickers'], config['research']['start_date'], now.strftime("%Y-%m-%d"))
            
            # 3. Inference (Snapshot at current Knowledge Time)
            batch = lab.snapshot(as_of=now, tickers=config['universe']['tickers'])
            
            if batch:
                # 4. Generate Actual Scores via Model
                with torch.no_grad():
                    # Handle both TorchScript and Python models
                    if hasattr(model, 'predict_dataset'):
                        scores = model.predict_dataset(batch).squeeze().numpy()
                    else:
                        # Direct TorchScript Inference
                        scores = model(batch.data['x_seq'], batch.data['x_spatial']).squeeze().numpy()
                
                # 5. Aggregation: Get LATEST score per unique ticker
                ticker_latest_scores = {}
                for i, ticker in enumerate(batch.tickers):
                    ticker_latest_scores[ticker] = float(scores[i])
                
                # 6. Build the Decile Ladder
                ladder = []
                for ticker, score in ticker_latest_scores.items():
                    ladder.append({"ticker": ticker, "score": score})
                ladder.sort(key=lambda x: x['score'], reverse=True)
                
                # 7. Risk Scaling (The Kelly Check)
                belief_score = 0.86 # This should ideally be dynamic
                
                # 8. OMS Execution
                target_weights = oms.calculate_target_weights(ladder, belief_score)
                oms.execute_rebalance(target_weights)
                
            else:
                print("⚠️ No data available for inference.")
                
        except Exception as e:
            print(f"❌ Paper Bot Loop Error: {e}")
            
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(run_paper_bot())
