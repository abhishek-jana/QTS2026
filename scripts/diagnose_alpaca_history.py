import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": api_secret
}

# Test Fetch for SPY in 2016
params = {
    "symbols": "SPY",
    "timeframe": "1Day",
    "start": "2016-01-01",
    "end": "2016-01-30",
    "adjustment": "raw",
    "feed": "iex"
}

url = "https://data.alpaca.markets/v2/stocks/bars"
resp = requests.get(url, headers=headers, params=params)

print(f"Status Code: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    bars = data.get('bars', {})
    if not bars:
        print("❌ ALPACA RETURNED ZERO BARS for 2016. This confirms a subscription/tier limit.")
    else:
        for ticker, ticker_bars in bars.items():
            print(f"✅ Success! Received {len(ticker_bars)} bars for {ticker}. First bar: {ticker_bars[0]['t']}")
else:
    print(f"❌ Error from Alpaca: {resp.text}")
