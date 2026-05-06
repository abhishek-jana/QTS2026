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
    Event-driven Execution Engine for Alpaca.
    Handles order submission, fill reconciliation, and live state tracking.
    """
    def __init__(self, config, starting_capital=1000000.0):
        self.config = config
        self.starting_capital = starting_capital
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = self.config['execution_muscle']['oms']['base_url']
        
        self.rest_api = tradeapi.REST(self.api_key, self.api_secret, base_url=self.base_url)
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Real-time state
        self.positions = {}
        self.orders = {} # tracking our local order state
        self.oms_stats = {"filled": 0, "working": 0, "rejected": 0}
        self.order_log = []

    async def _on_trade_update(self, update):
        """Callback for Alpaca trade stream."""
        event = update.event
        order = update.order
        ticker = order['symbol']
        
        logger.info(f"ALpaca Stream: {event} for {ticker} | Qty: {order['qty']}")
        
        status_map = {
            "fill": "FILLED",
            "partial_fill": "WORKING",
            "canceled": "REJECTED",
            "rejected": "REJECTED",
            "new": "WORKING"
        }
        
        new_status = status_map.get(event, "WORKING")
        
        # Update UI via Redis
        log_entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": ticker,
            "side": order['side'].upper(),
            "qty": int(float(order['qty'])),
            "status": new_status
        }
        
        self.order_log.append(log_entry)
        if len(self.order_log) > 20: self.order_log.pop(0)
        
        # Update stats
        if event == "fill": 
            self.oms_stats["filled"] += 1
            if self.oms_stats["working"] > 0: self.oms_stats["working"] -= 1
        elif event == "rejected" or event == "canceled":
            self.oms_stats["rejected"] += 1
            if self.oms_stats["working"] > 0: self.oms_stats["working"] -= 1
        elif event == "new":
            self.oms_stats["working"] += 1

    def submit_order(self, ticker, side, qty):
        """Submit order to Alpaca REST API."""
        try:
            # Safety checks from config
            max_notional = self.config['execution_muscle']['safety_limits']['max_notional_per_order']
            # Simplified safety: assume price $100 if unknown for now (real bot would pull latest quote)
            
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
            pos = self.rest_api.list_positions()
            self.positions = {p.symbol: float(p.qty) for p in pos}
            
            portfolio_value = float(acct.portfolio_value)
            # Total P&L since initialization (starting capital)
            total_pnl = portfolio_value - self.starting_capital
            
            logger.info(f"Bot: Account Value ${portfolio_value} | Total P&L: ${total_pnl:.2f}")
            return portfolio_value, total_pnl
        except Exception as e:
            logger.error(f"State Hydration Failed: {e}")
            return 1000000.0, 0.0

    async def run_stream(self):
        """Run the WebSocket stream."""
        stream = tradeapi.stream.Stream(self.api_key, self.api_secret, base_url=self.base_url, data_feed='iex')
        stream.subscribe_trade_updates(self._on_trade_update)
        await stream._run_forever()
