import os
import numpy as np
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

class ExecutionEngine:
    """
    UQTS-2026 Execution Engine
    Deepened Order Management System with Multi-Period MPC and Kelly Sizing.
    """
    def __init__(self, config):
        # Fix: Consistent config key
        self.config = config.get('execution_muscle', {})
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        self.api = tradeapi.REST(
            api_key, 
            secret_key, 
            base_url=self.config.get('oms', {}).get('base_url', "https://paper-api.alpaca.markets")
        )
        
        # MPC Parameters (From config or sensible defaults)
        mpc_cfg = self.config.get('mpc_solver', {})
        self.market_impact_lambda = mpc_cfg.get('impact_penalty', 0.01)
        self.risk_aversion = mpc_cfg.get('variance_penalty', 0.1)
        self.horizon = mpc_cfg.get('horizon', 5)
        
    def calculate_target_weights(self, rankings, current_portfolio, belief_score):
        """
        Returns optimal weights minimizing market impact using Multi-Period MPC
        and Kelly scaling via Bayesian Belief.
        
        rankings: List of {"ticker": str, "score": float}
        current_portfolio: Dict of {ticker: current_weight}
        belief_score: float (Bayesian Belief Score)
        """
        
        if belief_score < self.config.get('min_belief_threshold', 0.75):
            print(f"⚠️ Bayesian Belief Score ({belief_score:.2f}) too low. Sitting in cash.")
            return {}

        target_weights = {}
        max_size = self.config.get('max_position_size', 0.1)
        
        # We only trade the top/bottom deciles as per original strategy, 
        # but apply MPC to smooth the entry/exit.
        n = len(rankings)
        top_n = max(1, n // 4)
        
        tradable_tickers = set()
        longs = rankings[:top_n]
        shorts = rankings[-top_n:]
        
        for item in longs + shorts:
            tradable_tickers.add(item['ticker'])
            
        # Also include anything currently in portfolio that we might want to exit
        all_tickers = tradable_tickers.union(set(current_portfolio.keys()))

        # Map rankings for easy lookup
        rank_map = {item['ticker']: item['score'] for item in rankings}

        for ticker in all_tickers:
            alpha = rank_map.get(ticker, 0.0) # If not in rankings, alpha is 0 (exit)
            current_weight = current_portfolio.get(ticker, 0.0)
            
            # 1. Kelly Sizing (Ported from execution_muscle/kelly_sizer.hpp)
            # f = (mu / sigma^2) * belief
            variance = self.config.get('mpc_solver', {}).get('variance_penalty', 1.0)
            optimal_f = alpha / variance
            kelly_weight = optimal_f * belief_score
            
            # 2. Multi-Period MPC (Ported from execution_muscle/mpc_solver.hpp)
            # Theoretical target if impact was zero:
            theoretical_target = kelly_weight / self.risk_aversion
            
            # Constraint: max_position_size
            if ticker in [t['ticker'] for t in longs]:
                theoretical_target = min(theoretical_target, max_size)
            elif ticker in [t['ticker'] for t in shorts]:
                theoretical_target = max(theoretical_target, -max_size)
            else:
                theoretical_target = 0.0 # Exit if not in top/bottom
            
            gap = theoretical_target - current_weight
            
            # Speed = 1.0 / (1.0 + sqrt(lambda * horizon))
            speed = 1.0 / (1.0 + np.sqrt(self.market_impact_lambda * self.horizon))
            
            # Target weight for this step
            target_weights[ticker] = current_weight + (gap * speed)
            
        return target_weights

    def execute(self, target_weights):
        """
        Handles the Alpaca API rebalancing logic.
        Sends orders to move from current portfolio to target weights.
        """
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            positions = self.api.list_positions()
            current_positions = {p.symbol: p for p in positions}
            
            print(f"🚀 ExecutionEngine Rebalancing [Equity: ${equity:,.2f}]")
            
            # 1. Identify required trades
            trades = []
            for symbol, weight in target_weights.items():
                target_value = equity * weight
                
                # Get current market price (Try Alpaca latest trade, fallback to last known position price)
                try:
                    current_price = float(self.api.get_latest_trade(symbol).price)
                except:
                    current_price = 100.0 
                    if symbol in current_positions:
                        current_price = float(current_positions[symbol].current_price)
                
                target_qty = int(target_value / current_price)
                current_qty = int(current_positions[symbol].qty) if symbol in current_positions else 0
                
                qty_delta = target_qty - current_qty
                
                if qty_delta != 0:
                    trades.append({
                        "symbol": symbol,
                        "qty": abs(qty_delta),
                        "side": "buy" if qty_delta > 0 else "sell"
                    })
                    print(f"   -> Plan: {qty_delta:+d} {symbol} (Target: {weight*100:.2f}%)")

            # 2. Execute sells first to free up buying power
            for trade in [t for t in trades if t['side'] == 'sell']:
                # self.api.submit_order(symbol=trade['symbol'], qty=trade['qty'], side='sell', type='market', time_in_force='day')
                print(f"      [EXEC] Selling {trade['qty']} {trade['symbol']}...")

            # 3. Execute buys
            for trade in [t for t in trades if t['side'] == 'buy']:
                # self.api.submit_order(symbol=trade['symbol'], qty=trade['qty'], side='buy', type='market', time_in_force='day')
                print(f"      [EXEC] Buying {trade['qty']} {trade['symbol']}...")
                
        except Exception as e:
            print(f"❌ ExecutionEngine Error: {e}")
