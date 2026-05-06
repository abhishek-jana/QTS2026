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
        self.oms_stats = {"filled": 0, "working": 0, "rejected": 0}
        self.order_log = []
        
        # Legacy Stream Connection
        self.conn = tradeapi.StreamConn(
            self.api_key, 
            self.api_secret, 
            base_url=self.base_url,
            data_url=self.base_url.replace("api", "data") # Simple heuristic for data URL
        )

        # Register Legacy Handlers
        @self.conn.on(r'trade_updates')
        async def on_trade_updates(conn, channel, data):
            await self._handle_trade_update(data)

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
                "time": datetime.now().strftime("%H:%M:%S"),
                "ticker": ticker,
                "side": order['side'].upper(),
                "qty": int(float(order['qty'])),
                "status": new_status
            }
            
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
            portfolio_value = float(acct.portfolio_value)
            
            if self.starting_capital is None or self.starting_capital == 1000000.0:
                self.starting_capital = portfolio_value
                logger.info(f"Bot: Initialized starting capital to ${self.starting_capital:,.2f}")

            pos = self.rest_api.list_positions()
            self.positions = {p.symbol: float(p.qty) for p in pos}
            total_pnl = portfolio_value - self.starting_capital
            
            logger.info(f"Bot: Account Value ${portfolio_value:,.2f} | Total P&L: ${total_pnl:,.2f}")
            return portfolio_value, total_pnl
        except Exception as e:
            logger.error(f"State Hydration Failed: {e}")
            return self.starting_capital or 100000.0, 0.0

    async def check_market_status(self):
        """Check if the market is currently open."""
        try:
            clock = self.rest_api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Market Status Check Failed: {e}")
            return False

    async def run_stream(self):
        """Run the legacy WebSocket stream."""
        try:
            logger.info("Bot: Starting Legacy WebSocket Stream...")
            await self.conn.run(['trade_updates'])
        except Exception as e:
            logger.error(f"Alpaca Stream Error: {e}")
            await asyncio.sleep(5) # Backoff
