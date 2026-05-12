# PRD: V7.1 Architecture Refactor (Unified Intelligence)

## Objective
Transform the UQTS-2026 codebase from a monolithic, tightly-coupled script architecture into a modular, production-grade quantitative trading framework. This enables rapid experimentation with new models (e.g., GNNs, Options pricing) without modifying the core execution engines.

## Scope & Impact
This refactor affects how models are built, how configurations are loaded, and how the execution engines interact with data. It will **not** change the underlying mathematics or the verified +156% performance of the V7.4 Sniper; it only changes the *scaffolding* that holds it.

## Phased Implementation Plan

### Stage 1: Strict Interfaces (The Contracts)
We must define exactly what a module is expected to do before we can swap them out.
- **Action:** Create `qts_core/interfaces.py`.
- **Details:** Define Python `Protocols` for:
  - `IDataProvider`: Contract for getting Point-in-Time data.
  - `IAlphaBrain`: Contract for converting raw data into latent embeddings.
  - `IAlphaHead`: Contract for converting embeddings into specific predictions (e.g., Directional Alpha, Volatility).
  - `IExecutionSizer`: Contract for turning scores into portfolio weights.

### Stage 2: The Registry Pattern (Dynamic Loading)
We need a way to look up classes by a string name (from a config) rather than hardcoding `import SniperRanker`.
- **Action:** Create `qts_core/registry.py`.
- **Details:** Implement a robust Registry system with decorators.
  - `@register_brain("tft_v7")`
  - `@register_head("directional_log_return")`
  - `@register_plugin("wavelet_spatial")`

### Stage 3: The "Brains & Heads" Decoupling
We will tear apart the monolithic `SniperRanker`.
- **Action:** Refactor `research_lab/alpha_ranker_sniper.py`.
- **Details:** 
  - Extract the core TFT and ViT logic into a new `TFTBrain` class.
  - Extract the final linear layer into a `DirectionalAlphaHead` class.
  - Register both classes in the new Registry.

### Stage 4: Hierarchical Configuration (The Config Factory)
The single `config.yaml` is becoming a bottleneck. We need modular configurations.
- **Action:** Create a `configs/` directory structure and a config loader in `qts_core/config_loader.py`.
- **Details:**
  - `configs/main.yaml` (The master router)
  - `configs/models/sniper_v7.yaml` (Defines the Brain, Head, and hidden dims)
  - `configs/execution/live_paper.yaml` (Defines OMS and sizing rules)
  - The loader will merge these at runtime based on CLI arguments.

### Stage 5: The Engine Re-Wire (Factory Pattern)
The final step is to update the consumers to use the new modular system.
- **Action:** Refactor `alpha_factory/strategy_engine.py` and `execution_muscle/inference_worker.py`.
- **Details:**
  - Remove direct model imports.
  - Instantiate models dynamically: `self.brain = Registry.build_brain(config.model.brain_type)`
  - Instantiate execution sizers dynamically: `self.sizer = Registry.build_sizer(config.execution.sizer_type)`

### Stage 6: Comprehensive Testing (End-to-End Validation)
Because this is an end-to-end task, we must guarantee data integrity and module interaction.
- **Action:** Build a structured `tests/` directory covering all levels of the pipeline.
- **Details:**
  - **Data Ingestion (`tests/data/`)**: Validate DuckDB PiT logic and pipeline robustness to ensure the raw features are flawless before they hit the Brains.
  - **Unit Tests (`tests/unit/`)**: Verify `Registry` dynamic loading, `TFTBrain` embedding shapes, and `DirectionalAlphaHead` output bounds.
  - **Integration Tests (`tests/integration/`)**: Confirm the `StrategyEngine` successfully wires the config to the executing models, and the `InferenceWorker` generates correct payloads.

## Verification
- Run `uv run python run.py rl eval` after the refactor. 
- The system MUST produce the exact same +156% Alpha and -26% MDD results. If the numbers change, the refactor broke the mathematical integrity of the model.

## Migration & Rollback
- The work will be done on the `v7.1-unified-intelligence` branch.
- The `v7.0-sniper-residual` branch remains our stable, verified production fallback.
