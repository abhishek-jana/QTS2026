# PRD: Reinforcement Learning (RL) Portfolio Optimizer

## 1. The Analogy: "The ChatGPT of Trading"
- **Current State (The "GPT" base model)**: Your Multi-Modal RankNet is the world-class engine. It understands the "language" of the market and can predict returns (IC ~0.20), but it doesn't know how to *behave* in a portfolio.
- **Future State (The "ChatGPT" agent)**: By adding an RL layer, we teach the model a **Policy**. Instead of just predicting prices, it learns a **Strategy** for managing a $100k account by exploring millions of decisions to find the optimal path to wealth.

## 2. Technical Architecture: Hierarchical RL

The system will be split into two distinct layers:
1.  **Level 1: The Alpha Oracle (Challenger V2)**: Provides the Decile Ladder (Stock Rankings).
2.  **Level 2: The RL Pilot**: A custom agent that "drives" the account using the rankings as its input sensors.

### A. The Environment (The "Gym")
We will build a custom `PortfolioGym` using the **OpenAI Gymnasium** standard.
- **Observation Space (The Inputs)**:
    - **Alpha Scores**: Top 10 long scores and Bottom 10 short scores.
    - **Risk Metrics**: Current Bayesian Belief score, Portfolio Drawdown (%), and SPY 21-day realized volatility.
    - **Account State**: Current leverage, Buying Power, and Time-to-Rebalance.
- **Action Space (The Controls)**:
    - `Action 1`: **Gross Leverage** (Continuous: 0.0x to 2.5x).
    - `Action 2`: **Concentration** (Discrete: Top 2, Top 5, or Top 12 stocks).
    - `Action 3`: **Hedge Ratio** (Continuous: 0% to 50% short index ETFs).

### B. The Reward Function (The "Brain's North Star")
We **do not** reward purely on Profit (which leads to reckless gambling). Instead, we use the **Sortino Ratio**:
$$Reward = \frac{R_p - R_f}{\sigma_d}$$
*Where $R_p$ is portfolio return and $\sigma_d$ is the standard deviation of **downside** returns only.*
- **Penalty**: -10% reward penalty for hitting a margin call.
- **Penalty**: -5% reward penalty for exceeding -10% drawdown.

### C. The Algorithm: PPO (Proximal Policy Optimization)
We will use **PPO**, the same algorithm used by OpenAI to train ChatGPT.
- **Why PPO?** It is "stable." It prevents the agent from making radically different decisions between Monday and Tuesday, ensuring a smooth equity curve.

## 3. Training Protocol
1.  **Historical Rollouts**: The agent will "play" the 2018-2022 market data 10,000 times.
2.  **Random Start Dates**: Every training episode starts on a random day, so the agent doesn't just memorize the 2020 crash.
3.  **Adversarial Market**: We will occasionally "inject" random 5% market crashes into the data to force the agent to learn "Defensive Driving."

## 4. Expected Outcomes
- **Dynamic Sizing**: The agent learns to "Hammer" (2.5x leverage) when the top decile has a high spread and volatility is low.
- **Regime Switching**: The agent learns to go to **0.2x leverage** automatically when the Bayesian Belief is low, even if the model's scores look "okay."
- **Return Target**: **300% - 500%** cumulative gain over 3.5 years by optimizing the *timing* of leverage.

---
**Status**: DRAFT (Roadmap for Phase 3)
**Author**: Gemini CLI (Expert Quant)
