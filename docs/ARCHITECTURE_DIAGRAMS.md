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

## 6. The "Tri-Brain" Decoupled Logic
This diagram illustrates the operational separation of thinking (AI/RL) and execution (Bot). Each module is isolated to ensure system stability and prevent logic interference.

```mermaid
graph TD
    subgraph The Thinking Brain (The Cloud/Server)
        A[Analyst: RankNet AI] -- "Signal: 60-Stock Ladder" --> C[Inference Translation]
        B[Captain: RL Pilot] -- "Decision: 1.0x / Top 5 / Sniper" --> C
    end

    subgraph The Memory Layer (Persistence)
        C -- "Write Weights % (No thinking)" --> D[(Redis: Sealed Envelope)]
    end

    subgraph The Execution Muscle (Local/Paper)
        D -- "Read Weights % at 3:50 PM" --> E[Mechanic: PaperBot]
        F[Live Price Feed] -- "Get Penny Price" --> E
        E -- "Mechanical Order: (Weights * Budget) / Price" --> G[Broker: Alpaca/IBKR]
    end

    subgraph Operational Roles
        RoleA[Analyst: Sees Alpha, ignores dollars]
        RoleB[Captain: Sees profit/risk, ignores tickers]
        RoleC[Mechanic: Sees orders/penny prices, ignores strategy]
    end
```
