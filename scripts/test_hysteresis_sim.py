import asyncio
from execution_muscle.inference_worker import InferenceWorker
from datetime import datetime, timedelta
import numpy as np

async def run_fast_sim():
    worker = InferenceWorker()
    worker.trading_mode = 'sim'
    worker.update_interval = 0
    worker.initialize()
    
    # Disable redis
    class DummyRedis:
        def get(self, *args): return None
        def publish(self, *args): pass
        def smembers(self, *args): return []
    worker.redis_client = DummyRedis()
    
    start_time = datetime(2023, 1, 1, 16, 0, 0)
    end_time = datetime(2026, 5, 7, 16, 0, 0)
    worker.current_knowledge_time = start_time
    
    print(f"🚀 RUNNING HYSTERESIS + REGIME SIMULATION: {start_time.date()} -> {end_time.date()}")
    
    days_passed = 0
    total_val = 100000.0
    belief = 0.5
    
    while worker.current_knowledge_time <= end_time:
        worker._update_metacognition_feedback()

        house_view = worker.strategy.get_current_rankings(as_of=worker.current_knowledge_time, include_batch=True)
        if house_view['status'] == "OK":
            worker.ranking_history.append({
                'time': worker.current_knowledge_time,
                'rankings': {e['ticker']: e['score'] for e in house_view['ladder']}
            })
            
            belief = float(house_view['belief_score'])
            belief = max(0.05, min(0.95, belief))
            
            # Use the real OMS sim logic we just built
            worker._update_oms_sim(house_view, belief)
            
            # Ledger V2 accounting
            unrealized_pnl = 0.0
            current_mv = 0.0
            for ticker, qty in worker.sim_positions.items():
                current_p = worker._get_latest_price_sim(ticker) or worker.sim_avg_costs.get(ticker, 0)
                entry_p = worker.sim_avg_costs.get(ticker, 0)
                
                if qty > 0: # LONG
                    pnl = (current_p - entry_p) * qty
                    mv = current_p * qty
                else: # SHORT
                    pnl = (entry_p - current_p) * abs(qty)
                    mv = (entry_p * abs(qty)) + pnl
                
                unrealized_pnl += pnl
                current_mv += mv
            
            total_val = worker.starting_capital + worker.sim_realized_pnl + unrealized_pnl
            roi_pct = (total_val / worker.starting_capital) - 1.0
            worker.ls_equity_curve.append(roi_pct)
            if len(worker.ls_equity_curve) > 100: worker.ls_equity_curve.pop(0)

            days_passed += 1
            if days_passed % 100 == 0:
                print(f"[{worker.current_knowledge_time.date()}] Account: ${total_val:,.2f} | Belief: {belief*100:.1f}% | Active Pos: {len(worker.sim_positions)}")

        worker.current_knowledge_time += timedelta(days=1)
        
    print("\n" + "="*50)
    print("🏁 SIMULATION COMPLETE 🏁")
    print(f"Final Account Value: ${total_val:,.2f}")
    print(f"Total Return: {((total_val/worker.starting_capital)-1)*100:.2f}%")
    print(f"Final Bayesian Belief: {belief*100:.1f}%")
    print(f"Final Active Positions: {len(worker.sim_positions)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_fast_sim())
