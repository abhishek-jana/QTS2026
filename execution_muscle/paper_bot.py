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
        
    # 1. Initialize OMS & Research Lab (via DataStreamer for convenience)
    # Note: In a production crontab, we'd use a dedicated Orchestrator
    oms = OMS(config)
    
    # Placeholder for the DataStreamer to fetch PIT snapshots
    # (Simplified for the demo)
    from research_lab.alpha_universe import AlphaUniverse
    from research_lab.plugins.core_plugins import SequentialPlugin, SpatialPlugin
    from research_lab.real_data_ingestor import YFinanceIngestor
    from datetime import datetime
    
    lab = AlphaUniverse(plugins=[SequentialPlugin(), SpatialPlugin()])
    ingestor = YFinanceIngestor(lab.engine)
    
    while True:
        try:
            now = datetime.now()
            print(f"\n🕒 Loop Start: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 2. Ingestion (Latest Market Reality)
            ingestor.ingest_universe(config['universe']['tickers'], "2024-01-01", now.strftime("%Y-%m-%d"))
            
            # 3. Inference (Snapshot at current Knowledge Time)
            batch = lab.snapshot(as_of=now, tickers=config['universe']['tickers'])
            
            if batch:
                # 4. Scoring (Decile Ladder)
                ticker_scores = []
                for i, ticker in enumerate(batch.tickers):
                    ticker_scores.append({"ticker": ticker, "score": float(batch.labels[i])})
                ticker_scores.sort(key=lambda x: x['score'], reverse=True)
                
                # 5. Risk Scaling (The Kelly Check)
                belief_score = 0.86 # Mocked from Metacognition Panel
                
                # 6. OMS Execution
                target_weights = oms.calculate_target_weights(ticker_scores, belief_score)
                oms.execute_rebalance(target_weights)
                
            else:
                print("⚠️ No data available for inference.")
                
        except Exception as e:
            print(f"❌ Paper Bot Loop Error: {e}")
            
        # Run every hour for the demo, would be daily/intraday in production
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(run_paper_bot())
