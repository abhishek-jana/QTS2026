# Architectural Diagrams: UQTS-2026 Production-Grade

## 1. Data Ingestion & Storage Pipeline
```mermaid
graph LR
    A[Polygon.io OR Tiingo] -- "API / Unadjusted Ticks (adjusted=False)" --> B(InstitutionalIngestor: Python)
    B -- DataFrame Insertion --> C[(DuckDB: data/uqts_bitemporal.ddb)]
    C -- "SQL PIT View (Dynamic Columns)" --> D[AlphaUniverse: Python]
```

## 2. Research & Inference Pipeline (Quad-Modality)
```mermaid
graph TD
    A[(DuckDB)] -- "PIT Sliced Window" --> B[InferenceWorker: Python/PyTorch]
    
    subgraph Encoders
        B -- "Fractional Diff (Close)" --> C1[LSTM: Temporal Head]
        B -- "Mexican Hat CWT (Close)" --> C2[ViT: Spatial Head]
        B -- "Rolling Stationary State" --> C3[GNN: Relational Head]
        B -- "Log-Normalized Volume" --> C4[Sequential: Volume Head]
    end
    
    C1 & C2 & C3 & C4 -- "Feature Embeddings" --> D[Learned Modality Gating]
    D -- "Dynamically Weighted Fusion" --> E[RankNet Head]
    E -- "Pairwise Score" --> F[AlphaRanker: TorchScript]
    
    F -- Bayesian Belief --> G[MetaController: Python]
    G -- JSON Payloads --> H[(Redis: Pub/Sub)]
```

## 3. Execution Muscle Pipeline
```mermaid
graph TD
    A[(Redis: Pub/Sub)] -- Signal Update --> B[PaperBot: Python/Asyncio]
    B -- Covariance Matrix --> C[KellySizer: C++]
    B -- Alpha Decay vs Market Impact --> D[MPC Solver: C++/OSQP]
    D & C -- Optimal Weighting --> E[ExecutionEngine: C++]
    E -- Order Instructions --> F[Alpaca API: WebSockets]
    F -- Fill Updates --> G[Portfolio Reconciler: Asyncio]
```

## 4. UI Streaming Layer
```mermaid
graph TD
    A[(Redis: Pub/Sub)] -- Targeted Ticker Data --> B[FastAPI: Streamer]
    B -- Selective WebSockets --> C[Cockpit Frontend: React/LightweightCharts]
```
