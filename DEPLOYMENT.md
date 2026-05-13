# UQTS-2026: Institutional Live Pipeline Deployment SOP

This document provides the exact commands and operational schedule to run the **V7.4.3 "Patient Sniper"** execution framework end-to-end.

## 1. The Operational Schedule (EST Time)

The system follows a strict **T+1 Execution Model** to filter market noise and ensure maximum liquidity.

| Time Window | Phase | Description |
| :--- | :--- | :--- |
| **09:30 - 15:50** | **Monitoring** | Watch the live Decile Ladder and Signal Energy in the UI. |
| **15:50 - 16:00** | **Execution (T+1)**| Physical dispatch of MOC orders for *yesterday's* plan. |
| **16:05 - 16:30** | **Planning (T=0)** | AI ingests daily close and queues the plan for *tomorrow*. |
| **17:00 - 09:00** | **Off-Market** | Bot can be closed; state is persisted in Redis. |

---

## 2. End-to-End Pipeline (Daily Workflow)

### Step 1: Data Ingestion (Fresh Knowledge)
Run this daily after the 4:00 PM EST close to ensure the AI "Sword" is sharp.
```bash
python run.py signal ingest
```

### Step 2: Launch the Mission Control Dashboard
Start the backend and frontend to visualize the telemetry.
```bash
# Terminal Tab 1: Backend
python run.py ui

# Terminal Tab 2: UI (Open http://localhost:5173)
cd cockpit_frontend && npm run dev
```

### Step 3: Launch the Master Sniper (Live)
Start the execution worker. It will automatically recover its previous state from Redis.
```bash
python run.py live
```

---

## 3. Advanced Features for Solo Traders

### 🏎️ Self-Healing Boot (Retroactive Planning)
You do **not** need to keep your laptop running 24/7. 
*   If you boot the system at 10:00 PM (after the planning window), the bot will detect the missing plan.
*   It will automatically trigger **Retroactive Planning** to generate tomorrow's Strategy Queue immediately.
*   You can then review the plan and close the laptop until the next day's execution window.

### 🏁 State Recovery
The system persists the following data in Redis (`uqts:live:*`):
*   **Benchmarking History:** Performance charts survive restarts.
*   **SPY Anchor:** The $100k index baseline is locked to maintain Alpha accuracy.
*   **Sealed Envelope:** The pending trade decision is frozen until the execution window.

---

## 4. Emergency Procedures

### The Kill Switch
If you see unexpected behavior (e.g., extreme slippage or news-driven panic):
1.  Click the **`Kill_Sys`** button in the top-right of the UI.
2.  The worker will immediately set `is_killed = True` and attempt to cancel any working orders.
3.  Manually verify your account status in the Alpaca/Broker dashboard.

---
**Status:** V7.4.3 Institutional Blueprint Active.
**Strategy:** Shield & Sword (Decoupled Alpha/Beta).
