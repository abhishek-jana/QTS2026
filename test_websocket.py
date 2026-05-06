import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/ws/cockpit"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # 1. Wait for GLOBAL_UPDATE
            for i in range(10):
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Received message type: {data.get('type')}")
                if data.get('type') == 'GLOBAL_UPDATE':
                    print("✅ Successfully received GLOBAL_UPDATE")
                    
                    # 2. Test SET_TICKER command
                    subscribe_command = {
                        "command": "SET_TICKER",
                        "ticker": "NVDA"
                    }
                    await websocket.send(json.dumps(subscribe_command))
                    print("Sent SET_TICKER for NVDA")
                
                if data.get('type') == 'SPECTRAL_UPDATE':
                    print("✅ Successfully received SPECTRAL_UPDATE for NVDA")
                    break
            else:
                print("❌ Failed to receive expected updates within timeout")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
