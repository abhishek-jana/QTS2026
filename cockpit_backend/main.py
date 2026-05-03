from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from cockpit_backend.streamer import DataStreamer

app = FastAPI(title="UQTS-2026 Mission Control Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()
streamer = DataStreamer(manager)

@app.on_event("startup")
async def startup_event():
    # Start the background streamer
    asyncio.create_task(streamer.start_streaming())

@app.websocket("/ws/cockpit")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Handle incoming commands from UI (e.g. Kill Switch)
            if data == "KILL_SWITCH":
                print("⚠️ KILL SWITCH TRIGGERED FROM COCKPIT ⚠️")
                await manager.broadcast(json.dumps({"type": "ALERT", "msg": "EMERGENCY LIQUIDATION INITIATED"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health():
    return {"status": "operational", "clients": len(manager.active_connections)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
