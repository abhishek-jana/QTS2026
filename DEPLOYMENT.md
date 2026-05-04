# UQTS-2026: 1-Week Forward Testing Deployment SOP

This document provides the exact commands and procedures to initiate a 1-week autonomous paper trading trial on the **UQTS-2026** platform.

## 1. Prerequisites (Complete Tonight)

- [ ] **Alpaca Credentials**: Ensure `.env` contains:
  ```env
  ALPACA_API_KEY=your_key
  ALPACA_SECRET_KEY=your_secret
  ```
- [ ] **Global Config**: Review `config.yaml`.
  - Verify `universe.tickers` contains the stocks you want to trade (NVDA, META, etc.).
  - Verify `execution.min_belief_threshold` is set (default 0.75).
- [ ] **Dependency Sync**: Run `uv sync` to ensure the environment is locked.

## 2. Launching Persistent Session (Starting Tomorrow)

To ensure the bot runs even if your terminal closes or your laptop enters sleep mode, use a `tmux` session on your Linux workstation.

1. **Create a new session**:
   ```bash
   tmux new -s uqts-bot
   ```
2. **Configure Environment**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   ```
3. **Start the Bot**:
   ```bash
   uv run python -m execution_muscle.paper_bot
   ```

## 3. Persistent Operation Commands

- **To Detach**: Press `Ctrl + B`, then release and press `D`. The bot is now running in the background.
- **To Re-attach (Check Logs)**: 
  ```bash
  tmux attach -t uqts-bot
  ```
- **To Scroll Logs in Tmux**: Press `Ctrl + B`, then `[` (use arrow keys to scroll, press `Q` to exit scroll mode).

## 4. Market Cycle Behavior

- **Market Closed**: The bot logs `💤 Market is CLOSED` and checks every 15 minutes.
- **Market Open (09:31 AM EST)**: The bot initiates the loop:
  1. **Ingest**: Downloads latest quotes via `yfinance`.
  2. **Inference**: Generates RankNet Z-scores.
  3. **Risk Check**: Scales position sizes by Bayesian Belief.
  4. **Execute**: Calculates rebalance deltas and logs intent.
- **Portfolio Tracking**: Monitor live fills at [Alpaca Paper Web Dashboard](https://app.alpaca.markets/paper/dashboard/overview).

## 5. Mission Control UI

While the bot trades in the background, launch the dashboard for visual signal analysis:
```bash
# Terminal Tab 1
uv run python cockpit_backend/main.py

# Terminal Tab 2
cd cockpit_frontend && npm run dev
```

## 6. Emergency Liquidation

If a regime shift causes unexpected behavior:
1. **Re-attach**: `tmux attach -t uqts-bot`
2. **Stop Process**: `Ctrl + C`
3. **Liquidate**: Click the **Emergency Kill Switch** on the Mission Control UI or manually close all positions in the Alpaca Dashboard.

---
**Status**: STANDING BY FOR TOMORROW'S OPENING BELL.
**Strategy**: Signal vs. Fluid (Morlet CWT + RankNet).
