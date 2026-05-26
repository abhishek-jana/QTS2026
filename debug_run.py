from execution_muscle.inference_worker import InferenceWorker
import asyncio
from datetime import timedelta

async def test():
    worker = InferenceWorker(mode_override='sim')
    worker.rl_pilot = None
    worker.initialize()
    
    # Fast forward a few days
    for _ in range(5):
        dt_key = worker.current_knowledge_time.strftime("%Y-%m-%d")
        print("Checking day:", dt_key)
        hv = worker.sim_rankings_cache.get(dt_key, {"status": "MISSING", "ladder": []})
        if hv['status'] == "OK":
            for e in hv['ladder']: worker.sim_price_memory[e['ticker']] = e['price']
            stats = worker._update_oms_sim(hv)
            print("Stats:", stats)
            print("Positions:", worker.sim_positions)
        worker.current_knowledge_time += timedelta(days=1)

asyncio.run(test())
