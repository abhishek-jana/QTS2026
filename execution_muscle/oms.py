import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

class OMS:
    """
    Order Management System (OMS)
    Translates RankNet Z-scores into target weights and executes via Alpaca.
    """
    def __init__(self, config):
        self.config = config['execution']
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        self.api = tradeapi.REST(
            api_key, 
            secret_key, 
            base_url=self.config['alpaca']['base_url']
        )
        
    def calculate_target_weights(self, ranking_ladder, belief_score):
        """
        Converts the Decile Ladder into target portfolio weights.
        Applies Bayesian Belief scaling (the Kelly Check).
        """
        # If the brain is unconfident, sit in cash
        if belief_score < self.config['min_belief_threshold']:
            print(f"⚠️ Bayesian Belief Score ({belief_score:.2f}) too low. Sitting in cash.")
            return {}

        # 1. Select top 10% (Longs) and bottom 10% (Shorts)
        # For our 8-stock universe, we'll take top 2 and bottom 2.
        n = len(ranking_ladder)
        top_n = max(1, n // 4)
        
        longs = ranking_ladder[:top_n]
        shorts = ranking_ladder[-top_n:]
        
        # 2. Assign weights based on Z-score strength
        # Max 10% per stock, scaled by belief
        target_weights = {}
        scaling_factor = belief_score # Simple Kelly-like scaling
        
        for item in longs:
            target_weights[item['ticker']] = self.config['max_position_size'] * scaling_factor
            
        for item in shorts:
            target_weights[item['ticker']] = -self.config['max_position_size'] * scaling_factor
            
        return target_weights

    def execute_rebalance(self, target_weights):
        """
        Calculates deltas between current Alpaca portfolio and targets.
        Sends Limit Orders to bridge the gap.
        """
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            positions = self.api.list_positions()
            current_positions = {p.symbol: float(p.qty) for p in positions}
            
            print(f"🔄 OMS Rebalancing [Equity: ${equity:,.2f}]")
            
            for symbol, weight in target_weights.items():
                target_value = equity * weight
                # In a real environment, we'd fetch the latest bid/ask
                # and calculate the quantity. 
                # For now, we log the intent.
                print(f"   -> Target: {symbol} at {weight*100:.1f}% (${target_value:,.2f})")
                
            # Logic to handle sells first, then buys
            # [STUB] This would use self.api.submit_order()
            
        except Exception as e:
            print(f"❌ OMS Execution Error: {e}")
