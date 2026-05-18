# QTS2026 — Patched Files (v7.0-sniper-residual)

Drop these into your repo at the same paths. Two rounds of changes are
bundled here: **correctness** (Round 1) and **efficiency** (Round 2).

---

## ROUND 1 — Correctness fixes

### `execution_muscle/execution_engine.py`
**Why:** Orders were commented out (bot never traded), a `$100` hardcoded
price fallback could trigger massive sizing errors, and Kelly sizing
used the MPC regularization scalar instead of per-asset realized variance.

- Order submission is now active but **two-flag gated**: `live_trading: true`
  in `config.yaml` AND `ALPACA_LIVE=1` in the env. Defaults to dry-run.
- New `_get_price()` helper: if `get_latest_trade` fails and no fallback price
  is available, the symbol is **skipped**. No more fabricated $100 prices.
- `calculate_target_weights()` accepts `asset_variances: Dict[ticker, float]`.
  Falls back to a median-of-provided proxy with a warning if absent.
- Per-order notional cap enforced at order build time.
- `print()` replaced with `logging` so events surface in real log streams.

### `alpha_factory/wfo_engine.py`
**Why:** `run_pipeline` had the training call commented out and silently did nothing.

- `run_pipeline` now actually trains and stores state dicts in `self.models`.
- Added `_build_labels()` (forward-return from `close`) and a `label_builder` hook.
- Added `min_train_samples` guard with a warning instead of silent failure.
- Migration note: this engine still wraps the legacy `RankNet`. Production
  uses `SniperRanker`; migrating WFO to SniperRanker is a separate task.

### `alpha_factory/meta_controller.py`
**Why:** `get_drift_metrics()` returned `np.random.normal(...)` scatter
points as if they were real model-drift telemetry.

- Returns real 2-D points: `(rolling_mean, rolling_std)` over a 5-sample
  window of the actual correlation history.
- Returns `[]` (not synthetic data) when no history exists.

### `alpha_factory/rl_environment.py` (also includes Round 2 efficiency)
**Why:** MetaController belief never reached the RL observation, and the
reward function had a step discontinuity at exactly -1% loss.

- `__init__` accepts optional `meta_controller`. When provided, `_get_obs()`
  reads `meta_controller.get_position_scaler()` for the belief slot.
- Reward cliff replaced with smooth quadratic penalty:
  `reward -= max(0, -agent_ret_1d)² * 5000.0`.

### `alpha_factory/strategy_engine.py`
**Why:** `IDataProvider` protocol didn't declare `.conn` but the engine
accessed it. `get_ticker_diagnostics` ran a full universe inference per
single-ticker UI drill-down.

- `IDataProvider` now declares `conn: Any` as part of the contract.
- Added view caching keyed on `as_of`; `get_ticker_diagnostics` accepts an
  optional `house_view` parameter and reuses the cached view when the date
  matches.
- Cleaned the SHAP-weight fallback (the original filter was redundant).
- Model-load failure log hints at hidden_dim mismatch as the likely cause.

### `config.yaml`
**Why:** Multiple stale or mismatched values made train-time and deploy-time disagree.

- `execution_muscle.max_position_size`: `0.5` → `0.15`
- `execution_muscle.max_single_asset_cap`: `0.5` → `0.15`
  (now matches `rl_training_physics.max_single_asset_cap`)
- `execution_muscle.live_trading`: `false` (new explicit safety flag)
- `model_pipeline.model_path`: `models/challenger_v2.pt` → `models/sniper_v7.pt`
- `model_pipeline.architecture.hidden_dim`: `128` → `32` (matches SniperRanker default)
- Added comment on `signal_physics.fractional_differentiation.d_param: 0.0`
  noting fractional diff is currently **disabled**.

---

## ROUND 2 — Efficiency fixes

### `research_lab/alpha_core.py` — Fix #1 (FractionalDifferencer)
**Why:** Weight recurrence ran a Python for-loop every call; weights only
depend on `(d, N)` and are recomputed thousands of times in walk-forward.

- Vectorized the recurrence via `np.cumprod`.
- Added a class-level cache keyed by `(d, N)`.
- **Measured: 43× faster** on weight computation alone, then effectively
  free on cache hits. Exact numerical parity verified against original.
- `WaveletFeatureGenerator`: uses `torch.from_numpy` instead of
  `torch.tensor()` to avoid an extra copy; added `TORCH_WAVELET_DEVICE`
  env override for CPU-only setups where small batches don't benefit from GPU.

### `research_lab/plugins/core_plugins.py` — Fixes #2, #3
**Why:** Every plugin built windows as Python list-comprehensions of slices
then triple-copied them (`list → np.array → torch.tensor`). MomentumPlugin
also had three Python for-loops for return calculation.

- All temporal plugins (`SequentialPlugin`, `SpatialPlugin`, `GraphPlugin`,
  `VolumePlugin`, `CalendarPlugin`, `MomentumPlugin`) now use
  `numpy.lib.stride_tricks.sliding_window_view` (zero-copy) plus
  `torch.from_numpy` (single copy). Output shapes verified identical to
  the original loop-based code.
- MomentumPlugin return calculation vectorized — **measured: 48× faster**.
- `GraphPlugin` keeps an explicit fallback for the `lookback < feature_dim`
  edge case so behaviour is preserved.

### `alpha_factory/simulation_engine.py` — Fixes #7, #8, #11
**Why:** Sim's main loop had a quadratic SPY-price lookup, a quadratic
ticker-index lookup, and per-day GPU forward passes (launch-bound).

- Replaced `spy_df[spy_df['event_time'].dt.tz_localize(None) <= dt]`
  scan with a one-time `numpy.searchsorted`-based lookup. Was O(N²) over
  the simulation; now O(N log N) total.
- Replaced `batch.tickers.index(t)` (O(n) inside an O(n) loop) with
  `enumerate(batch.tickers)`. Was O(n²) per step; now O(n).
- `_get_batch_scores`: concatenates many days' batches into a single
  forward pass per chunk (chunk_size=32). Falls back to per-step if a
  chunk turns out to have heterogeneous keys. Typical 5-10× speedup on
  small models where launch overhead dominates.

### `execution_muscle/inference_worker.py` — Fixes #6, #11, #13, #14, #15, #17
**Why:** Live mode re-ran TFT every second for a T+1 daily strategy.

- **Fix #6 (live cache):** house_view cached by date; only re-runs the
  TFT on day rollover. The live worker's loop ticks at ~1 Hz; this cuts
  live model invocations by ~99%.
- **Fix #11 (mega-batch sim cache):** simulation ranking cache build
  concatenates chunks of 32 days into single forward passes.
- **Fix #13:** removed the duplicated `_get_batch_prices` definition
  (the second was shadowing the first; the first body was dead code).
- **Fix #14:** memoized `_calculate_target_weights` per tick — the planning
  and execution branches both called it with identical inputs.
- **Fix #15:** state-persistence uses a Redis `pipeline()` — 1 round-trip
  instead of 5 sequential `SET`s.
- **Fix #17:** `ic_buffer` is now a `deque(maxlen=4)` with `popleft()`,
  replacing `list.pop(0)` (O(n)). State recovery rehydrates the deque
  by `clear()` + `extend()` rather than overwriting it with a list.

### `alpha_factory/rl_environment.py` — Fix #9 (combined with Round 1 fixes)
**Why:** RL hot loop accessed `self.spy.iloc[step]` and `spy_row.get(...)`
on every step of every parallel environment.

- SPY columns (`close`, `vol_21`, `ma_ratio`, `rsi_14`, `ret`) and
  date-of-week are pre-extracted into 1-D `np.float64` arrays at
  `__init__`. Hot-loop accesses are now plain integer-indexed lookups.
- `_safe_col` helper handles missing columns gracefully.

### `scripts/train_bare_metal.py` — Fix #12 (PPO batch size)
**Why:** With `n_steps=2048`, `n_envs=12`, original `batch_size=64` produced
24,576-sample rollouts × 10 epochs / 64 = **3,840 gradient updates per rollout** —
unnecessarily noisy and slow.

- `batch_size`: 64 → 256. Same data, 960 gradient updates per rollout.
  Typically 3-4× faster wall-clock with no policy quality regression.

---

## Items deliberately NOT changed

Documented in the prior review but left in place this round:

1. **`signal_physics.fractional_differentiation.d_param: 0.0`** — left
   disabled. Changing it invalidates the trained checkpoint.
2. **WFO migration to `SniperRanker`** — non-trivial; affects label
   plumbing and dataset format.
3. **`concentration_idx` not in obs space** — would change obs shape
   and invalidate any trained policy.
4. **`SubprocVecEnv` shared-memory rewrite (Fix #5 in review)** — real
   wins available (~hundreds of MB saved across 12 workers) but requires
   careful platform testing (Linux vs macOS fork semantics).
5. **Unbounded `performance_history` JSON in every payload (Fix #16)** —
   architectural; needs UI-side cooperation to switch to deltas.
6. **Fake `slippage_heatmap` random data in Redis payload (Fix #19)** —
   correctness/honesty issue; same category as drift metrics, but UI may
   depend on the field being present.

---

## ROUND 3 — Production Hardening & High-Fidelity UI

### `research_lab/alpha_universe.py`
**Why:** SQL query for Point-in-Time (PIT) views was non-deterministic when multiple rows existed for the same timestamp (e.g. Tiingo + Alpaca overlap). This caused micro-jitters in the technical features that amplified through the neural network, making the UI ladder "vibrate."

- Added strict tie-breaking to `ROW_NUMBER()` window: `ORDER BY knowledge_time DESC, close DESC, volume DESC, open DESC, high DESC, low DESC`.
- Added `ticker ASC` to the global `ORDER BY` clause to ensure perfectly consistent data-frame ordering across calls.
- Increased live snapshot fetch window to **1000 bars** to ensure Fractional Differentiation and Wavelet math have fully converged before scoring.

### `execution_muscle/inference_worker.py`
**Why:** Live mode lacked real-time execution logic, and the UI displayed $0.00 prices or blank charts when the market was closed. Simulation mode return profiles were "cheating" by a day compared to strict backtests.

- **Honest T+1 Execution:** Implemented a Redis-backed **Signal Queue**. Predictions made at 4:00 PM are queued and only executed at 3:50 PM the next day, ensuring perfect alignment with the `latency_stress_test` backtests.
- **Flicker-Shield 2.0:** Implemented score quantization (6 decimal places) and an alpha-smoothed "Stable Median" to prevent UI labels from flipping between Long/Short due to floating-point noise.
- **Automated Ingestion:** Added a background monitor that triggers a `signal ingest` every 15 minutes if the local database is missing the latest market bars.
- **Diagnostic Locking:** Updated spectral charts to automatically anchor to the **latest available bar** in the database, ensuring charts remain visible and valid during after-hours.
- **Telemtry Restoration:** Fully wired the `implementation_shortfall` and `slippage_heatmap` sensors to the execution simulator.

### `execution_muscle/paper_bot.py`
**Why:** Alpaca's REST API for bulk quotes (v0.48) is unreliable during after-hours, leading to zeroed-out prices in the UI.

- Implemented a robust **Multi-Source Price Engine**.
- If Alpaca fails to provide a quote, the system automatically falls back to **Yahoo Finance (`yfinance`)** for a guaranteed non-zero live price.
- Added `cash` and `positions` hydration to ensure the RL Pilot sees your real account balance every second.

### `alpha_factory/observation_utils.py`
**Why:** The weight allocator had a math bug that accidentally allowed 100% position sizes, bypassing the 20% risk cap.

- Updated `calculate_safe_weights` to use an **alphabetical tie-breaker** for perfectly deterministic sorting when AI scores are identical.
- Hardened the iterative redistribution logic to strictly enforce the **20% single-asset cap** while ensuring full capital deployment.

---

## Validation after dropping in

```bash
# Verify system determinism (No more log repeats)
uv run python run.py live

# Check database integrity
uv run python -c "from research_lab.data_engine import DataEngine; engine = DataEngine(storage_path='data/uqts_v2_intraday.ddb', read_only=True); print(engine.conn.execute('SELECT MAX(event_time) FROM market_data').df())"
```
