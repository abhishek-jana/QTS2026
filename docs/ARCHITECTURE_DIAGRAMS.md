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

## 3. The "Shield & Sword" Architecture (V7.4.3)
```mermaid
graph TD
    subgraph Sword: Alpha Engine (RankNet)
        A[60-Ticker Universe] -- "63-Day PIT Window" --> B[Temporal Fusion Transformer]
        B -- "Cross-Sectional Ranking" --> C[Top 5 Picks]
    end
    
    subgraph Shield: Macro Risk (RL Pilot)
        D[32-Sensor Observation] -- "VIX, Vol_Vel, RSI, Drawdown" --> E[PPO Policy Pilot]
        E -- "Exposure Decision" --> F[0.0x or 1.0x Leverage]
    end
    
    C & F -- "Fused Intelligence" --> G[Strategy Queue: T+1 Plan]
    G -- "JSON Serialize" --> H[(Redis: pending_decision)]
```

## 4. T+1 Execution Muscle Pipeline
```mermaid
graph TD
    A[4:15 PM EST: T=0 Close] --> B[Generate Strategy Queue]
    B -- "Freeze & Persist" --> C[(Redis: Sealed Envelope)]
    
    C -- "Boot-up Recover" --> D[InferenceWorker: Python/Asyncio]
    
    D -- "Wait for MOC Window" --> E{3:50 PM EST: T+1?}
    E -- YES --> F[OMS: Order Management System]
    F -- "MOC Orders" --> G[Broker: Alpaca/IBKR]
    
    G -- "Fill @ 4:00 PM Close" --> H[Portfolio Reconciler]
    H -- "Update History" --> I[(Redis: State Recovery)]
```

## 5. UI Streaming Layer
```mermaid
graph TD
    A[(Redis: Pub/Sub)] -- Targeted Ticker Data --> B[FastAPI: Streamer]
    B -- Selective WebSockets --> C[Cockpit Frontend: React/LightweightCharts]
```
