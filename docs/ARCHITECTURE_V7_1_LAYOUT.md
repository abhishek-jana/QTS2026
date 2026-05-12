# Target Architecture Layout: V7.1 (Unified Intelligence)

To support the modular decoupling and dynamic registry pattern, the V7.1 codebase will be restructured into the following directory tree. This layout emphasizes strict institutional nomenclature and modularity, ensuring that new models, tasks, or execution rules can be added without modifying the core engines.

## Directory Tree

```text
QTS2026/
│
├── configs/                             <-- Hierarchical Configuration
│   ├── main.yaml                        <-- Master router (points to specific sub-configs)
│   ├── models/
│   │   ├── sniper_v7_4.yaml             <-- Defines TFT Encoder + Directional Predictor
│   │   └── experimental_gnn.yaml        <-- (Future) Defines GNN Encoder
│   ├── execution/
│   │   ├── prod.yaml                    <-- Live trading rules, sizing caps, paper trader settings
│   │   └── sim.yaml                     <-- Backtesting execution constraints
│   └── environments/
│       └── rl_v7_4.yaml                 <-- Defines RL action space, panic thresholds, rewards
│
├── core/                                <-- Core System Infrastructure (Formerly qts_core)
│   ├── __init__.py
│   ├── interfaces.py                    <-- Strict Python Protocols (IEncoder, IPredictor, IDataProvider)
│   ├── registry.py                      <-- Component Registry (@register_encoder, @register_predictor)
│   ├── config_loader.py                 <-- Merges YAML files dynamically at runtime
│   └── logger.py                        <-- System logging
│
├── data/                                <-- Dedicated Data Engineering Pillar
│   ├── __init__.py
│   ├── ingestors/                       <-- Extensible multi-vendor ingestion
│   │   ├── __init__.py
│   │   ├── tiingo_ingestor.py           <-- (Active) Tiingo API fetching logic
│   │   ├── alpaca_ingestor.py           <-- (Future) Alpaca market data
│   │   └── binance_ingestor.py          <-- (Future) Crypto tick data
│   └── engine.py                        <-- DuckDB PiT serving logic (implements IDataProvider)
│
├── research/                            <-- Representation & Task Learning (Formerly research_lab)
│   ├── __init__.py
│   ├── encoders/                        <-- Core Representation Models (Formerly Brains)
│   │   ├── __init__.py
│   │   └── tft_encoder.py               <-- Extracts features into dense vectors
│   ├── predictors/                      <-- Task-Specific Output Layers (Formerly Heads)
│   │   ├── __init__.py
│   │   └── directional_predictor.py     <-- Converts embeddings to log-return predictions
│   ├── plugins/
│   │   └── core_plugins.py              <-- Wavelet, spatial, calendar transforms
│   └── universe.py                      <-- Walk-forward engine
│
├── strategy/                            <-- Orchestration (Formerly alpha_factory)
│   ├── __init__.py
│   ├── engine.py                        <-- Uses Registry to instantiate Encoder+Predictor
│   ├── rl_environment.py                <-- Sniper V7.4 gym
│   ├── simulation.py                    <-- Backtesting orchestrator
│   └── meta_controller.py               <-- Bayesian drift tracking
│
├── execution/                           <-- Execution & Sizing (Formerly execution_muscle)
│   ├── __init__.py
│   ├── inference_worker.py              <-- Consumes hierarchical config
│   ├── sizers/
│   │   ├── __init__.py
│   │   └── risk_parity.py               <-- Softmax & conviction scaling
│   └── paper_trader.py                  <-- Live OMS simulation (Formerly paper_bot)
│
├── scripts/                             
│   ├── train_rl_policy.py               <-- (Formerly train_rl_pilot)
│   └── evaluate_policy.py               <-- (Formerly rl_evaluator)
│
├── tests/                               <-- End-to-End & Modular Testing
│   ├── unit/
│   │   ├── test_registry.py             <-- Tests dynamic loading of encoders/predictors
│   │   ├── test_encoders.py             <-- Tests embedding generation shapes/types
│   │   └── test_predictors.py           <-- Tests prediction outputs
│   ├── integration/
│   │   ├── test_strategy_engine.py      <-- Tests the Factory wiring config -> engine
│   │   └── test_inference_worker.py     <-- Tests mock payload generation
│   └── data/
│       └── test_data_ingestion.py       <-- Tests DuckDB PiT logic and pipeline robustness
│
└── run.py                               <-- Entrypoint CLI (now uses ConfigLoader)
```

## Key Files to be Created/Refactored:

1. **`core/interfaces.py`**: Will define `Protocol` classes (`IEncoder`, `IPredictor`, `IExecutionSizer`) ensuring type-safety across the platform.
2. **`core/registry.py`**: A Singleton or module-level dictionary managing `@register` decorators.
3. **`core/config_loader.py`**: A utility using `OmegaConf` or deep-merge logic to combine `main.yaml` with the referenced sub-configs.
4. **`research/encoders/tft_encoder.py`**: Extracted from the monolithic model.
5. **`research/predictors/directional_predictor.py`**: Extracted from the monolithic model.
6. **`strategy/engine.py`**: Stripped of hardcoded imports, replaced with `Registry.build_encoder(...)`.

**Status:** Awaiting execution.
