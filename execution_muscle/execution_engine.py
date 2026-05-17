import os
import logging
import numpy as np
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    UQTS-2026 Execution Engine
    Deepened Order Management System with Multi-Period MPC and Kelly Sizing.

    Safety design:
      - `live_trading` MUST be explicitly set True in config AND
        ALPACA_LIVE=1 must be in the env. Default is dry-run (no orders).
      - If the latest price for a symbol cannot be retrieved, the symbol is
        SKIPPED (not sized with a fake $100). Bad data must never produce
        a market order.
      - Per-asset variance is required for true Kelly sizing. If the caller
        does not supply one, we fall back to a portfolio-level proxy but
        emit a warning so the operator can wire real volatilities.
    """

    def __init__(self, config):
        self.config = config.get('execution_muscle', {})
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        self.api = tradeapi.REST(
            api_key,
            secret_key,
            base_url=self.config.get('oms', {}).get('base_url', "https://paper-api.alpaca.markets"),
        )

        # MPC Parameters (from config or sensible defaults)
        mpc_cfg = self.config.get('mpc_solver', {})
        self.market_impact_lambda = mpc_cfg.get('impact_penalty', 0.01)
        self.risk_aversion = mpc_cfg.get('variance_penalty', 0.1)
        self.horizon = mpc_cfg.get('horizon', 5)

        # SAFETY: order submission is OFF unless explicitly enabled in BOTH
        # config and environment. Belt-and-braces to prevent accidental live trading.
        env_live = os.getenv("ALPACA_LIVE", "0") == "1"
        cfg_live = bool(self.config.get('live_trading', False))
        self.live_trading = env_live and cfg_live
        if self.live_trading:
            logger.warning("ExecutionEngine: LIVE trading ENABLED. Orders will be submitted.")
        else:
            logger.info("ExecutionEngine: dry-run mode. No orders will be submitted.")

    def calculate_target_weights(
        self,
        rankings,
        current_portfolio,
        belief_score,
        asset_variances=None,
    ):
        """
        Returns optimal weights using Multi-Period MPC smoothing and Kelly
        scaling via Bayesian Belief.

        rankings: List of {"ticker": str, "score": float}
        current_portfolio: Dict of {ticker: current_weight}
        belief_score: float (Bayesian Belief Score)
        asset_variances: Optional Dict of {ticker: realized return variance}.
                         If absent, a global proxy is used and a warning logged.
        """
        if belief_score < self.config.get('min_belief_threshold', 0.75):
            logger.warning(
                f"Bayesian belief score ({belief_score:.2f}) below threshold — sitting in cash."
            )
            return {}

        if asset_variances is None:
            logger.warning(
                "calculate_target_weights: no asset_variances supplied; "
                "Kelly sizing will use a constant proxy. Pass realized "
                "variances per asset for proper sizing."
            )
            asset_variances = {}

        # Global proxy variance: median of provided, or a conservative default
        # (~20% annualized vol squared, scaled to daily ~ 0.04/252)
        if asset_variances:
            proxy_var = float(np.median(list(asset_variances.values())))
        else:
            proxy_var = 0.04 / 252.0

        target_weights = {}
        max_size = self.config.get('max_position_size', 0.1)

        n = len(rankings)
        top_n = max(1, n // 4)
        longs = rankings[:top_n]
        shorts = rankings[-top_n:]

        long_tickers = {t['ticker'] for t in longs}
        short_tickers = {t['ticker'] for t in shorts}
        rank_map = {item['ticker']: item['score'] for item in rankings}
        all_tickers = long_tickers | short_tickers | set(current_portfolio.keys())

        for ticker in all_tickers:
            alpha = rank_map.get(ticker, 0.0)  # not in rankings → exit
            current_weight = current_portfolio.get(ticker, 0.0)

            # 1. Kelly: f* = mu / sigma^2, scaled by belief
            var_t = asset_variances.get(ticker, proxy_var)
            var_t = max(var_t, 1e-6)
            kelly_weight = (alpha / var_t) * belief_score

            # 2. MPC: theoretical target under no impact
            theoretical_target = kelly_weight / self.risk_aversion

            # Hard cap by side
            if ticker in long_tickers:
                theoretical_target = min(theoretical_target, max_size)
            elif ticker in short_tickers:
                theoretical_target = max(theoretical_target, -max_size)
            else:
                theoretical_target = 0.0  # exit-only

            gap = theoretical_target - current_weight
            speed = 1.0 / (1.0 + np.sqrt(self.market_impact_lambda * self.horizon))
            target_weights[ticker] = current_weight + (gap * speed)

        return target_weights

    def _get_price(self, symbol, current_positions):
        """Return latest trade price or None if unavailable. NEVER fabricates."""
        try:
            return float(self.api.get_latest_trade(symbol).price)
        except Exception as e:
            logger.warning(f"Latest trade fetch failed for {symbol}: {e}")

        pos = current_positions.get(symbol)
        if pos is not None:
            try:
                return float(pos.current_price)
            except Exception:
                pass
        return None  # caller must skip this symbol

    def execute(self, target_weights):
        """
        Compute deltas vs current Alpaca positions and submit orders.
        Submissions are gated by self.live_trading.
        """
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            positions = self.api.list_positions()
            current_positions = {p.symbol: p for p in positions}

            mode = "LIVE" if self.live_trading else "DRY-RUN"
            logger.info(f"ExecutionEngine [{mode}] rebalancing | Equity: ${equity:,.2f}")

            trades = []
            for symbol, weight in target_weights.items():
                target_value = equity * weight
                price = self._get_price(symbol, current_positions)
                if price is None or price <= 0:
                    logger.error(
                        f"Skipping {symbol}: no valid price available. "
                        f"Refusing to size order with unknown price."
                    )
                    continue

                target_qty = int(target_value / price)
                current_qty = int(current_positions[symbol].qty) if symbol in current_positions else 0
                qty_delta = target_qty - current_qty

                # Safety: notional cap per order
                max_notional = self.config.get('safety_limits', {}).get(
                    'max_notional_per_order', 50000.0
                )
                if abs(qty_delta) * price > max_notional:
                    logger.warning(
                        f"{symbol}: order notional ${abs(qty_delta)*price:,.0f} exceeds cap "
                        f"${max_notional:,.0f}. Truncating."
                    )
                    qty_delta = int(np.sign(qty_delta) * (max_notional / price))

                if qty_delta != 0:
                    trades.append({
                        "symbol": symbol,
                        "qty": abs(qty_delta),
                        "side": "buy" if qty_delta > 0 else "sell",
                    })
                    logger.info(
                        f"   Plan: {qty_delta:+d} {symbol} @ ${price:,.2f} "
                        f"(target {weight*100:.2f}%)"
                    )

            # Sells first to free buying power
            for trade in [t for t in trades if t['side'] == 'sell']:
                self._submit(trade)
            for trade in [t for t in trades if t['side'] == 'buy']:
                self._submit(trade)

        except Exception as e:
            logger.error(f"ExecutionEngine error: {e}")

    def _submit(self, trade):
        if not self.live_trading:
            logger.info(f"   [DRY-RUN] would {trade['side']} {trade['qty']} {trade['symbol']}")
            return
        try:
            self.api.submit_order(
                symbol=trade['symbol'],
                qty=trade['qty'],
                side=trade['side'],
                type='market',
                time_in_force='day',
            )
            logger.info(f"   [EXEC] {trade['side']} {trade['qty']} {trade['symbol']} submitted.")
        except Exception as e:
            logger.error(f"   [EXEC] {trade['symbol']} order failed: {e}")
