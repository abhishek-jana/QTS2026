import asyncio
import json
import redis
import yaml
from qts_core.logger import logger

class DataStreamer:
    """
    Consumer-side Streamer. 
    Listens to Redis and multiplexes to WebSockets.
    """
    def __init__(self, manager):
        self.manager = manager
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            logger.info("DATASTREAMER: Connected to Redis successfully.")
        except Exception as e:
            logger.error(f"DATASTREAMER: Redis connection failed: {e}")
            
        self.is_initialized = False
        self.is_killed = False

    async def initialize(self):
        self.is_initialized = True
        logger.info("✅ DATASTREAMER: Initialization Complete")

    def handle_command(self, websocket, data):
        """Handle UI commands."""
        if isinstance(data, str) and data == 'KILL_SWITCH':
            logger.warning("🚨 KILL SWITCH ACTIVATED VIA UI")
            # Publish to worker
            self.redis_client.publish('uqts:commands', json.dumps({"command": "KILL_SWITCH"}))
            return

        if isinstance(data, dict) and data.get("command") == "SET_TICKER":
            ticker = data.get("ticker")
            self.manager.subscribe(websocket, ticker)
            # Store in Redis so InferenceWorker knows what to compute
            self.redis_client.sadd('uqts:watchlist', ticker)
            logger.info(f"🎯 UI Focus changed to: {ticker}")

    async def start_streaming(self):
        """Listen to Redis channels and broadcast."""
        # Use pattern subscribe to catch global and spectral updates
        self.pubsub.psubscribe('uqts:*')
        logger.info("🚀 DATASTREAMER: Redis Listener Loop Started")
        
        while not self.is_killed:
            message = self.pubsub.get_message(ignore_subscribe_messages=True)
            if message and message['type'] == 'pmessage':
                channel = message['channel']
                data = message['data']
                
                if channel == 'uqts:global':
                    await self.manager.broadcast(data)
                elif channel.startswith('uqts:spectral:'):
                    ticker = channel.split(':')[-1]
                    await self.manager.send_to_subscribers(ticker, data)
            
            await asyncio.sleep(0.01) # High frequency

    def kill_switch(self):
        self.is_killed = True
