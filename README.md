<p align="center">
  <img src="assets/logo.svg" alt="Causal Uplift Engine Logo" width="130" height="130">
</p>

<h1 align="center">Causal Uplift & Retention Engine</h1>
<p align="center">
  <strong>X-Learner & Doubly Robust CATE for Optimal Retention Intervention</strong><br>
  <em>Meta-learners estimating individual treatment effects and solving constrained Knapsack budget allocation.</em>
</p>

<p align="center">
  <a href="https://github.com/Rangga11268/causal-uplift-retention-engine/actions/workflows/ci.yml"><img src="https://github.com/Rangga11268/causal-uplift-retention-engine/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python Version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/architecture-zero--docker%20%7C%20cross--platform-emerald.svg" alt="Architecture"></a>
</p>

Production-grade Causal Inference and Uplift Modeling engine designed to eliminate marketing budget waste in customer retention and promotion allocation. Rather than predicting raw churn probability $P(Y=1 \mid X)$, this engine estimates the **Conditional Average Treatment Effect (CATE)** using state-of-the-art meta-learners (X-Learner, Doubly Robust Learner) and solves the optimal intervention policy via Constrained Knapsack Optimization under fixed daily marketing budgets.

![Causal Qini Cumulative Gains & Uplift by Decile](demo_qini_retention.png)

---

## 1. Problem Statement: Why Churn Prediction Bleeds Budget

Standard predictive churn models identify users with a high likelihood of leaving. In commercial operations (*Gojek, Grab, Tokopedia, Shopee, Uber*), offering promotional vouchers blindly to predicted churners causes severe margin degradation because customer responses follow four distinct causal archetypes:

```
                         Treatment (Promo Voucher)
                   ┌───────────────────┬───────────────────┐
                   │   Not Received    │     Received      │
┌──────────────────┼───────────────────┼───────────────────┤
│ Persuadables     │ Churns (Y=0)      │ Converts (Y=1)    │ -> TARGET (High Positive CATE)
│ Sure Things      │ Converts (Y=1)    │ Converts (Y=1)    │ -> DO NOT TARGET (Zero Uplift / Waste)
│ Lost Causes      │ Churns (Y=0)      │ Churns (Y=0)      │ -> DO NOT TARGET (Zero Uplift / Waste)
│ Sleeping Dogs    │ Converts (Y=1)    │ Churns (Y=0)      │ -> AVOID (Negative CATE / Destructive)
└──────────────────┴───────────────────┴───────────────────┘
```

Only **Persuadables** generate incremental revenue. Causal Uplift Modeling quantifies the individual treatment effect:

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]$$

---

## 2. Mathematical Foundations & Causal Meta-Learners

### 2.1 Propensity Score Estimation with Positivity Guards
To control for observational selection bias, propensity scores $e(x) = P(T=1 \mid X=x)$ are estimated and strictly bounded away from the extremes:

$$e(x) = \text{clip}\left(P(T=1 \mid X=x), \epsilon, 1 - \epsilon\right) \quad \text{where } \epsilon = 0.01$$

### 2.2 X-Learner (Künzel et al., PNAS 2019)
Specially designed for imbalanced treatment ratios common in production A/B tests and observational marketing data:
1. **First Stage (Response Estimation):**
   Fit $\hat{\mu}_0(x)$ on control $(X_0, Y_0)$ and $\hat{\mu}_1(x)$ on treated $(X_1, Y_1)$.
2. **Second Stage (Counterfactual Imputation):**
   $$D_{1,i} = Y_{1,i} - \hat{\mu}_0(X_{1,i}) \quad (\text{Imputed effect on treated})$$
   $$D_{0,j} = \hat{\mu}_1(X_{0,j}) - Y_{0,j} \quad (\text{Imputed effect on control})$$
3. **Third Stage (Effect Regression):**
   Fit effect regressors $\hat{\tau}_1(X_1) \to D_1$ and $\hat{\tau}_0(X_0) \to D_0$.
4. **Fourth Stage (Adaptive Propensity Weighting):**
   $$\hat{\tau}_{\text{X}}(x) = e(x)\hat{\tau}_0(x) + (1 - e(x))\hat{\tau}_1(x)$$

### 2.3 Doubly Robust (DR) Learner (Chernozhukov & Semenova, 2021)
Applies Neyman-orthogonal score projection with $K$-fold cross-fitting to remove regularization biases:

$$\Gamma_i = \left(\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)\right) + \frac{T_i\left(Y_i - \hat{\mu}_1(X_i)\right)}{e(X_i)} - \frac{(1 - T_i)\left(Y_i - \hat{\mu}_0(X_i)\right)}{1 - e(X_i)}$$

$$\hat{\tau}_{\text{DR}} = \arg\min_{\tau} \sum_{i=1}^N \left(\Gamma_i - \tau(X_i)\right)^2$$

### 2.4 Causal Validation: Cumulative Qini Curve & AUUC
Because counterfactual ground-truth cannot be observed simultaneously for a single user, models are evaluated by ranking customers descending by predicted $\hat{\tau}(x)$:

$$Q(u) = Y_1(u) - \frac{N_1(u)}{N_0(u)} Y_0(u)$$

$$\text{AUUC} = \int_0^1 Q(u) \, du$$

$$\text{Qini Score} = \text{AUUC}_{\text{Model}} - \text{AUUC}_{\text{Random}}$$

### 2.5 Constrained Knapsack Budget Allocation
Given a candidate pool of $N$ users, Customer Lifetime Value $V_i$, intervention costs $c_i$, and total budget $B$, the policy solves:

$$\max_{\mathbf{z} \in \{0, 1\}^N} \sum_{i=1}^N z_i \cdot \hat{\tau}(x_i) \cdot V_i \quad \text{subject to} \quad \sum_{i=1}^N z_i \cdot c_i \le B$$

Using causal efficiency density sorting ($\text{Density}_i = \frac{\hat{\tau}(x_i) \cdot V_i}{c_i}$) with strict exclusion of non-positive responders ($\hat{\tau}(x_i) \le 0$).

---

## 3. Benchmark Results

Evaluated on Python 3.12 (CPU, 10,000 customer observational retention population, 70/30 train-test split):

| Metric | Target Specification | Achieved Result | Evaluation Method |
| :--- | :---: | :---: | :--- |
| **Qini Score (AUUC Gain)** | $> +10.0$ over Random | **+18.68** (AUUC 137.70 vs 119.02) | `demo.py` on held-out test set |
| **Top Decile Uplift (D1)** | $\ge +25.0\%$ | **+42.0%** incremental lift | Decile uplift table |
| **Sleeping Dogs Filtered** | Negative responders excluded | **100% Excluded** from targeting | `app/core/optimizer.py` |
| **Net ROI vs Naïve Churn** | $\ge +25.0\%$ Relative Gain | **+240.9% Net ROI** (vs +3.4% Naïve) | Simulated Knapsack allocation under $4,000 budget |
| **10k Users Knapsack Speed** | $< 50.0\text{ ms}$ | **8.60 ms** | `benchmarks/benchmark_uplift.py` |
| **Scoring Latency** | $< 50.0\text{ \mu s}$ / customer | **13.28 \mu s / customer** (66.4 ms / 5k) | Batch inference benchmark |
| **Automated Test Suite** | 100% Pass Rate | **13 / 13 Passed (100%)** | `pytest tests/` |

> [!NOTE]
> **Experimental Context & Simulation Assumptions:**
> * **Simulation Scope:** The $+240.9\%$ Net ROI metric reflects a controlled counterfactual simulation on a 10,000-customer synthetic retention population where the Data Generating Process (DGP) contains known ground-truth treatment effects across 4 customer archetypes (Persuadables, Sure Things, Lost Causes, and Sleeping Dogs).
> * **Knapsack Economic Parameters:** The optimization assumes a fixed campaign budget $B = \$4,000$, intervention cost $c = \$15$ per user contacted, and retained customer lifetime value $V = \$100$. The causal policy sorts by density $\frac{\hat{\tau}(x) \cdot V}{c}$ and strictly filters Sleeping Dogs ($\hat{\tau}(x) \le 0$), whereas the naive churn baseline targets customers based solely on high churn probability without considering treatment sensitivity.
> * **Production Deployment:** In live production environments without unobserved confounding, treatment effects must be estimated from historical randomized A/B experiments or observational data satisfying unconfoundedness ($Y(1), Y(0) \perp T \mid X$) with sufficient overlap.

---

## 4. Architecture & Repository Structure

```
causal-uplift-retention-engine/
├── .github/workflows/
│   └── ci.yml                     # Multi-OS (Ubuntu & Windows, Py 3.10-3.12) CI
├── app/
│   ├── api/
│   │   ├── routes.py              # FastAPI endpoints (/health, /v1/estimate-cate, /v1/optimize-allocation)
│   │   └── schemas.py             # Pydantic telemetry input/output contracts
│   ├── core/
│   │   ├── meta_learners.py       # S-Learner, T-Learner, X-Learner, DR-Learner
│   │   ├── metrics.py             # Qini curves, AUUC, and Decile Uplift tables
│   │   ├── optimizer.py           # Constrained Knapsack budget optimization
│   │   └── propensity.py          # Calibrated propensity score estimators
│   ├── data/
│   │   └── generator.py           # Synthetic observational retention stream with 4 archetypes
│   ├── viz/
│   │   └── qini_renderer.py       # Matplotlib Qini curves and decile bar charts
│   └── config.py                  # Hyperparameters and business budget constants
├── benchmarks/
│   └── benchmark_uplift.py        # Profiling scoring latency and knapsack solver
├── tests/
│   ├── test_api.py                # REST API endpoint verification
│   ├── test_meta_learners.py      # CATE estimation consistency & recovery tests
│   ├── test_metrics.py            # Qini monotonicity & AUUC boundary checks
│   └── test_optimizer.py          # Budget adherence & causal pruning tests
├── demo.py                        # Single-command end-to-end interactive simulation
├── demo_qini_retention.png        # Publication-ready Qini and Decile visualization
├── requirements.txt               # Pure Python dependencies (zero-Docker, CPU native)
└── README.md
```

---

## 5. Quickstart Guide (Zero-Docker / Cross-Platform)

### 5.1 Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Linux and Windows)

### 5.2 Installation
```bash
git clone https://github.com/Rangga11268/causal-uplift-retention-engine.git
cd causal-uplift-retention-engine

# Create and activate virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install lightweight dependencies
pip install -r requirements.txt
```

### 5.3 Run Interactive Showcase Demo
Simulates 10,000 customers, fits meta-learners, evaluates Qini curves, solves Knapsack budget allocation, and saves visualization:
```bash
python demo.py
```

### 5.4 Run Automated Test Suite
```bash
python -m pytest tests/ -v
```

### 5.5 Run Latency & Scaling Benchmarks
```bash
python benchmarks/benchmark_uplift.py
```

### 5.6 Launch FastAPI Decision Service
```bash
uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at `http://localhost:8000/docs`.

---

## 6. References & Literature

1. **Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B.** (2019). *Metalearners for estimating heterogeneous treatment effects using machine learning.* **Proceedings of the National Academy of Sciences (PNAS)**, 116(10), 4156–4165.
2. **Semenova, V., & Chernozhukov, V.** (2021). *Debiased Machine Learning of Conditional Average Treatment Effects and Other Causal Functions.* **The Econometrics Journal**, 24(2), 264–289.
3. **Radcliffe, N. J., & Surry, P. D.** (2011). *Real-World Uplift Modelling with Significance-Based Uplift Trees.* Stochastic Solutions White Paper.
4. **Gutierrez, P., & Gérardy, J. Y.** (2017). *Causal Inference and Uplift Modelling: A Review of the Literature.* JMLR Workshop and Conference Proceedings, 67, 1–13.
