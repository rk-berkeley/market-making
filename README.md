# Asian Option Pricing & Optimal Market-Making under Fad Dynamics

> **Quantitative Finance Research Project** — INDENG 222, Spring 2026  
> UC Berkeley, Industrial Engineering & Operations Research

---

## Overview

This repository contains both the theoretical derivations and full numerical implementation for two interconnected problems in quantitative finance:

1. **Asian Option Pricing** in the Black-Scholes framework via Monte Carlo simulation and variance-reduction techniques.
2. **Optimal Market-Making under Fad Price Dynamics** — a stochastic control problem where asset prices temporarily deviate from fundamental value due to latent "fad" effects, with both informed and uninformed trader populations.

The market-making model is based on the recent paper:  
> Barucci, Mathieu & Sánchez-Betancourt (2025), *"Market Making with Fads, Informed, and Uninformed Traders,"* [arXiv:2501.03658](https://arxiv.org/abs/2501.03658)

---

## Key Results

### Part 1 — Asian Option Pricing

| Result | Details |
|---|---|
| **Price ordering** | Proved and verified: $C^{\text{geom}}_{\text{Asian}} \le C^{\text{arith}}_{\text{Asian}} \le C_{\text{BS}}$ for all 100 parameter combinations |
| **Variance reduction** | Geometric averaging consistently narrows 95% CIs by 2–10% depending on moneyness |
| **Convexity** | Numerical confirmation: min second finite difference $> 0$ across full strike grid |
| **Monte Carlo** | $M = 200{,}000$ paths, $n = 50$ steps, exact log-normal increments (no discretisation bias) |

### Part 2 — Optimal Market-Making

| Result | Details |
|---|---|
| **Spread invariance** | $\delta^{a,*} + \delta^{b,*} = 2/k - 2A(t)$ — spread is **independent of fad and informed-trader share** |
| **Adverse selection via skew** | Informed flow enters exclusively through quote asymmetry $\delta^{a,*} - \delta^{b,*}$, not spread widening |
| **Information value** | FI outperforms PI by ~0.6% in mean objective; both dominate CJP as fad weight $q \to 1$ |
| **Kalman-Bucy** | Filter closes FI–PI gap as $q \to 1$, consistent with the separation principle |

| Strategy | Mean Objective | Std Dev | Mean Spread |
|---|---|---|---|
| Full Information (FI) | 21.33 | 4.94 | 2.0930 |
| Partial Information (PI) | 21.20 | 4.95 | 2.0930 |
| CJP Benchmark | 21.18 | 4.94 | 2.0930 |

---

## Repository Structure

```
.
├── asian_options/
│   ├── black_scholes.py     # Analytical BS pricing, Greeks, convexity check
│   ├── monte_carlo.py       # Exact GBM simulation, arithmetic/geometric MC pricer
│   └── plots.py             # Figure generation for Q1–Q4
│
├── market_making/
│   ├── model.py             # Model parameters, state dynamics, intensity functions
│   ├── hjb_solver.py        # Riccati ODE (A), linear ODEs (B, C), optimal controls
│   ├── kalman_filter.py     # Kalman-Bucy filter for partial-information estimation
│   └── simulation.py        # Monte Carlo benchmark: FI vs. PI vs. CJP
│
├── notebooks/
│   └── simulation.ipynb     # Full end-to-end Jupyter notebook
│
├── figures/                 # Pre-compiled PDF figures (see below)
├── report/
│   ├── main.tex             # Full LaTeX report with proofs
│   └── references.bib
│
├── requirements.txt
└── README.md
```

---

## Mathematical Framework

### Part 1: Asian Options

The risk-neutral price of an arithmetic Asian call is:
$$\text{CallAsian}(r, \sigma, S_0, T, K) = e^{-rT} \mathbb{E}^{\mathbb{Q}}\!\left[\left(\frac{1}{T}\int_0^T S_t \, dt - K\right)^+\right]$$

No closed-form exists for the arithmetic average of log-normals, motivating Monte Carlo. The geometric average $A_T^{\text{geom}} = \exp\!\left(\frac{1}{n+1}\sum \ln S_{t_i}\right)$ is used as a control variate, exploiting $A_T^{\text{geom}} \le A_T^{\text{arith}}$ path-wise via the AM-GM inequality.

### Part 2: Market-Making Model

**State dynamics:**
$$dS_t = \mu \, dt + \sigma \, dW_t, \qquad dU_t = -\eta U_t \, dt + dB_t$$

where $W_t = p Z_t + q U_t$ decomposes the price Brownian motion into a fundamental component ($Z$) and a fad component ($U$, an OU process).

**Order arrival intensities** (informed + uninformed traders):
$$\lambda^a_t = \left(\phi e^{-k\delta^a} + \psi e^{-k\delta^a - \gamma(\sigma q U_t \vee S^-)}\right) \mathbf{1}_{\{Q_{t-} > \underline{q}\}}$$

**Value function ansatz** (quadratic in inventory):
$$V(t, q, u) = q^2 A(t) + q B(t, u) + C(t, u)$$

This reduces the HJB jump-diffusion PDE to a **Riccati ODE** for $A(t)$ (closed-form) and **linear ODEs** for $B$, $C$. The optimal feedback controls are:
$$\delta^{a,*}(t,q,u) = \frac{1}{k} + (2q-1)A(t) + B(t,u)$$

**Kalman-Bucy filter** (partial information):
$$d\hat{U}_t = -\eta \hat{U}_t \, dt + \sigma^{-1}(q - \eta q \hat{P}_t) \, dI_t$$

---

## Figures

| Figure | Description |
|---|---|
| `fig_q1_convexity.pdf` | BS call price convexity in strike $K$ |
| `fig_q3_ci_length.pdf` | 95% CI width: arithmetic vs. geometric averaging |
| `fig_q4_comparison.pdf` | Price ordering: Geometric ≤ Arithmetic ≤ European |
| `fig_displacements.pdf` | Optimal ask/bid displacements as functions of fad $u$ |
| `fig_filter.pdf` | Kalman-Bucy filter vs. true fad for $q \in \{0.3, 0.6, 0.9\}$ |
| `fig_gamma_paths.pdf` | Inventory paths under different informed-flow sensitivity $\gamma$ |
| `fig_perf_q.pdf` | Strategy performance vs. fad weight $q$ |
| `fig_perf_gamma.pdf` | Strategy performance vs. informed-flow sensitivity $\gamma$ |
| `fig_perf_eta.pdf` | Strategy performance vs. mean-reversion speed $\eta$ |
| `fig_informed_share.pdf` | Spread, skew, and performance vs. informed-trader share |
| `fig_histograms.pdf` | Terminal objective distributions: FI vs. PI vs. CJP |
| `fig_coefficients.pdf` | ODE coefficients $A(t)$, $b_0(t)$, $b_1(t)$ over time |

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Asian option pricing scripts

```python
from asian_options import call_price, asian_option_mc, confidence_interval_95

# Black-Scholes European call
price = call_price(S0=50, K=50, r=0.05, sigma=0.30, T=1)
print(f"BS call price: {price:.4f}")

# Arithmetic Asian call with 95% CI
pa, sa = asian_option_mc(50, 50, 0.05, 0.30, 1, method="arithmetic")
_, _, width = confidence_interval_95(pa, sa, M=200_000)
print(f"Asian call: {pa:.4f}  (95% CI width: {width:.4f})")
```

### Run the market-making simulation

```python
from market_making import ModelParams, run_benchmark

params  = ModelParams(M_paths=3000)
results = run_benchmark(params)

for strategy, stats in results.items():
    print(f"{strategy}: mean = {stats['mean_obj']:.2f}, spread = {stats['mean_spread']:.4f}")
```

### Jupyter notebook

```bash
jupyter notebook notebooks/simulation.ipynb
```

---

## Theory References

| Reference | Role in this project |
|---|---|
| Kemna & Vorst (1990), *J. Banking & Finance* | Asian option pricing and geometric average control variate |
| Avellaneda & Stoikov (2008), *Quant. Finance* | Classical HFT/market-making stochastic control baseline |
| Cartea, Jaimungal & Penalva (2015), *Cambridge UP* | CJP benchmark strategy and general algorithmic trading framework |
| Barucci, Mathieu & Sánchez-Betancourt (2025), arXiv:2501.03658 | Primary model: fad dynamics, informed/uninformed traders, HJB solution |
| Bain & Crisan (2009), *Springer* | Kalman-Bucy filtering theory |
| Shreve (2004), *Springer Finance* | Stochastic calculus foundations |

---

## Topics Covered

- **Stochastic Calculus**: Itô's formula, Girsanov theorem, Doléans-Dade exponential
- **Monte Carlo Methods**: Exact GBM simulation, variance reduction, CLT-based confidence intervals
- **Stochastic Control**: Hamilton-Jacobi-Bellman (HJB) equations, viscosity solutions
- **Optimal Control**: Riccati ODEs, quadratic value function ansatz, Pontryagin maximum principle
- **Stochastic Filtering**: Kalman-Bucy filter, separation principle, innovation processes
- **Market Microstructure**: Bid-ask spread, adverse selection, informed trading, inventory management
- **Mathematical Finance**: Risk-neutral pricing, Asian options, Black-Scholes model

---

*Full technical derivations, proofs, and numerical results are in [`report/main.tex`](report/main.tex).*
