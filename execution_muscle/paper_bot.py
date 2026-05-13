import asyncio
import json
import yaml
import os
import numpy as np
from datetime import datetime
import alpaca_trade_api as tradeapi
from qts_core.logger import logger
import redis

class AsyncPaperBot:
    """
    Event-driven Execution Engine for Alpaca (Legacy v0.48 Compatible).
    Handles order submission, fill reconciliation, and live state tracking.
    """
    def __init__(self, config, starting_capital=None):
        self.config = config
        self.starting_capital = starting_capital 
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = self.config['execution_muscle']['oms']['base_url']
        
        self.rest_api = tradeapi.REST(self.api_key, self.api_secret, base_url=self.base_url)
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Real-time state
        self.positions = {}
        self.position_avg_costs = {} # Track entry prices
        self.position_unrealized_pnl = {} # Current P&L %
        self.buying_power = 0.0
        self.portfolio_value = 0.0
        self.oms_stats = {"filled": 0, "working": 0, "rejected": 0}
        self.order_log = []
        self.conn = None

    async def _handle_trade_update(self, data):
        """Reconciles fills from legacy stream data."""
        try:
            event = data.event
            order = data.order
            ticker = order['symbol']
            
            logger.info(f"Alpaca Stream: {event} for {ticker}")
            
            status_map = {
                "fill": "FILLED",
                "partial_fill": "WORKING",
                "canceled": "REJECTED",
                "rejected": "REJECTED",
                "new": "WORKING"
            }
            
            new_status = status_map.get(event, "WORKING")
            
            log_entry = {
                "time": datetime.now().strftime("%m/%d %H:%M:%S"),
                "ticker": ticker,
                "side": order['side'].upper(),
                "qty": int(float(order['qty'])),
                "price": float(order.get('filled_avg_price', 0)) or float(order.get('limit_price', 0)) or 0.0,
                "status": new_status
            }
            # Calculate notional for the log
            log_entry["notional"] = log_entry["qty"] * log_entry["price"]
            
            self.order_log.append(log_entry)
            if len(self.order_log) > 20: self.order_log.pop(0)
            
            if event == "fill": 
                self.oms_stats["filled"] += 1
                if self.oms_stats["working"] > 0: self.oms_stats["working"] -= 1
            elif event in ["rejected", "canceled"]:
                self.oms_stats["rejected"] += 1
                if self.oms_stats["working"] > 0: self.oms_stats["working"] -= 1
            elif event == "new":
                self.oms_stats["working"] += 1
        except Exception as e:
            logger.error(f"Stream Handling Error: {e}")

    def submit_order(self, ticker, side, qty):
        """Submit order to Alpaca REST API."""
        try:
            self.rest_api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side.lower(),
                type='market',
                time_in_force='day'
            )
            logger.info(f"Order Submitted: {side} {qty} {ticker}")
        except Exception as e:
            logger.error(f"Order Submission Failed: {e}")
            self.oms_stats["rejected"] += 1

    async def hydrate_state(self):
        """Sync positions and cash from exchange."""
        logger.info("Bot: Synchronizing state with Alpaca...")
        try:
            acct = self.rest_api.get_account()
            self.portfolio_value = float(acct.portfolio_value)
            self.buying_power = float(acct.buying_power)
            
            # Initial setup of starting capital for P&L tracking
            if self.starting_capital is None:
                self.starting_capital = self.portfolio_value
                logger.info(f"Bot: Initialized starting capital to ${self.starting_capital:,.2f}")

            pos = self.rest_api.list_positions()
            self.positions = {p.symbol: float(p.qty) for p in pos}
            self.position_avg_costs = {p.symbol: float(p.avg_entry_price) for p in pos}
            self.position_unrealized_pnl = {p.symbol: float(p.unrealized_plpc) * 100.0 for p in pos}
            
            total_pnl = self.portfolio_value - self.starting_capital
            
            logger.info(f"Bot: Portfolio Value ${self.portfolio_value:,.2f} | Buying Power: ${self.buying_power:,.2f}")
            return self.portfolio_value, total_pnl
        except Exception as e:
            logger.error(f"State Hydration Failed: {e}")
            return self.starting_capital or 100000.0, 0.0

    def calculate_order_qty(self, ticker, price):
        """Calculates dynamic share quantity based on config limits and buying power."""
        if price <= 0: return 0
        
        limits = self.config['execution_muscle']
        # Unification: Prefer asset cap over legacy position size
        max_pos_pct = limits.get('max_single_asset_cap', limits.get('max_position_size', 0.15))
        max_notional = limits.get('safety_limits', {}).get('max_notional_per_order', 10000.0)
        
        # Target notional based on portfolio value
        target_notional = self.portfolio_value * max_pos_pct
        
        # Clip by safety limits
        target_notional = min(target_notional, max_notional)
        
        # Ensure we don't exceed available buying power
        # Note: In Alpaca, buying power for non-margin is usually portfolio_value
        target_notional = min(target_notional, self.buying_power * 0.95) # 5% buffer
        
        qty = int(target_notional / price)
        if qty > 0:
            logger.info(f"Sizer: Calculated {qty} shares for {ticker} (Notional: ${qty*price:,.2f})")
        return qty

    async def check_market_status(self):
        """Check if the market is currently open."""
        try:
            clock = self.rest_api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Market Status Check Failed: {e}")
            return False

    async def run_stream(self):
        """Run the legacy WebSocket stream in a dedicated thread with its own loop to avoid clashes."""
        import threading
        
        def _stream_worker():
            # Create a private event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Initialize Connection INSIDE the private loop
                self.conn = tradeapi.StreamConn(
                    self.api_key, 
                    self.api_secret, 
                    base_url=self.base_url,
                    data_url=self.base_url.replace("api", "data")
                )

                # Register Legacy Handlers
                @self.conn.on(r'trade_updates')
                async def on_trade_updates(conn, channel, data):
                    await self._handle_trade_update(data)

                logger.info("Bot: Alpaca Stream Thread (Isolated) started.")
                # We call run() which might be sync or async depending on SDK version
                coro = self.conn.run(['trade_updates'])
                if asyncio.iscoroutine(coro):
                    loop.run_until_complete(coro)
            except Exception as e:
                logger.error(f"Alpaca Stream Thread Error: {e}")
            finally:
                loop.close()
                logger.info("Bot: Alpaca Stream Thread closed.")

        try:
            logger.info("Bot: Launching Legacy WebSocket Stream (Thread-Isolated)...")
            threading.Thread(target=_stream_worker, daemon=True).start()
        except Exception as e:
            logger.error(f"Alpaca Stream Launch Error: {e}")
            await asyncio.sleep(5) # Backoff
