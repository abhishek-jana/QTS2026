from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import sys
import os
from contextlib import asynccontextmanager

# Ensure project root is in path
sys.path.append(os.getcwd())

from cockpit_backend.streamer import DataStreamer

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, str] = {} # WebSocket -> Subscribed Ticker

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        # Default focus ticker to SPY on connect
        self.active_connections[websocket] = 'SPY'
        print(f"📡 NEW CLIENT CONNECTED: {websocket.client.host if websocket.client else 'unknown'}")
        print(f"📊 ACTIVE CONNECTIONS: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]
        print(f"📡 CLIENT DISCONNECTED. REMAINING: {len(self.active_connections)}")

    def subscribe(self, websocket: WebSocket, ticker: str):
        if websocket in self.active_connections:
            self.active_connections[websocket] = ticker
            print(f"🔔 Client subscribed to focus ticker: {ticker}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Broadcast error: {e}")

    async def send_to_subscribers(self, ticker: str, message: str):
        """Sends only to clients subscribed to this specific ticker."""
        for connection, sub_ticker in self.active_connections.items():
            if sub_ticker == ticker:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    print(f"Send to subscriber error: {e}")

manager = ConnectionManager()
streamer = DataStreamer(manager)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Start the initialization task (non-blocking)
    init_task = asyncio.create_task(streamer.initialize())
    
    # 2. Start the background streamer
    stream_task = asyncio.create_task(streamer.start_streaming())
    
    yield
    
    # Clean up tasks on shutdown
    init_task.cancel()
    stream_task.cancel()

app = FastAPI(title="UQTS-2026 Mission Control Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/cockpit")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for commands
            data = await websocket.receive_text()
            
            try:
                # 1. Parse JSON commands (Dynamic Ticker Selection, etc.)
                command_data = json.loads(data)
                if "command" in command_data:
                    streamer.handle_command(websocket, command_data)
            except json.JSONDecodeError:
                # 2. Handle legacy string commands (Kill Switch)
                if data == "KILL_SWITCH":
                    print("⚠️ KILL SWITCH TRIGGERED FROM COCKPIT ⚠️")
                    streamer.kill_switch()
                    await manager.broadcast(json.dumps({"type": "ALERT", "msg": "EMERGENCY LIQUIDATION INITIATED"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health():
    return {"status": "operational", "clients": len(manager.active_connections)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
