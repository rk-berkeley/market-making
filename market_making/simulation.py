"""
simulation.py
-------------
Monte Carlo simulation engine: runs M paths under three strategies and
collects performance statistics.

Strategies
----------
FI  : Full Information  — market maker observes U_t directly
PI  : Partial Information — market maker uses Kalman-Bucy filter Û_t
CJP : Cartea-Jaimungal-Penalva benchmark — ignores fad entirely (q=0, gamma=0)

Reference: Barucci, Mathieu & Sánchez-Betancourt (2025), arXiv:2501.03658.
"""

import numpy as np
from typing import Literal
from market_making.model import (
    ModelParams,
    simulate_fad,
    simulate_price,
    order_intensities,
    performance_objective,
)
from market_making.hjb_solver import solve_hjb_coefficients, optimal_displacements
from market_making.kalman_filter import kalman_riccati, kalman_filter_path


# ---------------------------------------------------------------------------
# CJP benchmark displacements
# ---------------------------------------------------------------------------

def _cjp_displacements(
    t_idx: int,
    q: int,
    A: np.ndarray,
    b0: np.ndarray,
    params: ModelParams,
) -> tuple[float, float]:
    """Cartea-Jaimungal-Penalva strategy: ignore fad, u is treated as 0."""
    B  = b0[t_idx]       # b1 * 0 = 0 since fad is ignored
    d_a = 1.0 / params.k + (2 * q - 1) * A[t_idx] + B
    d_b = 1.0 / params.k - (2 * q + 1) * A[t_idx] - B
    d_a = max(d_a, -params.delta_inf)
    d_b = max(d_b, -params.delta_inf)
    return d_a, d_b


# ---------------------------------------------------------------------------
# Single-path simulation
# ---------------------------------------------------------------------------

def _simulate_one_path(
    strategy: Literal["FI", "PI", "CJP"],
    params: ModelParams,
    A: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    P_vals: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Simulate one path and return (objective, spread_avg, X_T, Q_T)."""
    # --- Simulate state processes ---
    U_path = simulate_fad(params, rng)
    S_path = simulate_price(params, U_path, rng)

    # --- Choose signal variable ---
    if strategy == "FI":
        signal = U_path
    elif strategy == "PI":
        signal = kalman_filter_path(S_path, P_vals, params, rng)
    else:  # CJP
        signal = np.zeros_like(U_path)

    # --- Market maker dynamics ---
    Q = 0
    X = 0.0
    spreads = []

    for i in range(params.N_steps):
        u   = signal[i]
        S_i = S_path[i]
        dt  = params.dt

        # Compute optimal displacements
        if strategy in ("FI", "PI"):
            d_a, d_b = optimal_displacements(i, Q, u, A, b0, b1, params)
        else:
            d_a, d_b = _cjp_displacements(i, Q, A, b0, params)

        spreads.append(d_a + d_b)

        # Stochastic intensities
        lam_a, lam_b = order_intensities(d_a, d_b, u, Q, params)

        # Poisson fills (thin-event approximation: at most 1 fill per dt)
        fill_a = rng.random() < lam_a * dt and Q > params.q_min
        fill_b = rng.random() < lam_b * dt and Q < params.q_max

        if fill_a:
            X += S_i + d_a
            Q -= 1
        if fill_b:
            X -= S_i - d_b
            Q += 1

    inventory_path = np.zeros(params.N_steps + 1)  # simplified: not tracked per step
    obj = performance_objective(X, Q, S_path[-1], inventory_path, params)
    return obj, np.mean(spreads), X, float(Q)


# ---------------------------------------------------------------------------
# Full Monte Carlo benchmark
# ---------------------------------------------------------------------------

def run_benchmark(params: ModelParams) -> dict:
    """Run M paths under FI, PI, and CJP strategies and compute statistics.

    Returns
    -------
    dict with keys "FI", "PI", "CJP", each mapping to:
        mean_obj   : float  -- mean realised objective
        std_obj    : float  -- standard deviation of realised objective
        mean_spread: float  -- mean time-averaged spread
    """
    rng = np.random.default_rng(params.seed)

    # Pre-compute ODE coefficients once
    t_grid, A, b0, b1 = solve_hjb_coefficients(params)
    P_vals = kalman_riccati(params)

    results = {s: {"objs": [], "spreads": []} for s in ("FI", "PI", "CJP")}

    for _ in range(params.M_paths):
        for strategy in ("FI", "PI", "CJP"):
            obj, spread, *_ = _simulate_one_path(
                strategy, params, A, b0, b1, P_vals, rng
            )
            results[strategy]["objs"].append(obj)
            results[strategy]["spreads"].append(spread)

    summary = {}
    for s in ("FI", "PI", "CJP"):
        objs = np.array(results[s]["objs"])
        summary[s] = {
            "mean_obj":    objs.mean(),
            "std_obj":     objs.std(),
            "mean_spread": np.mean(results[s]["spreads"]),
        }
    return summary


if __name__ == "__main__":
    params  = ModelParams(M_paths=500)     # small M for quick test
    summary = run_benchmark(params)
    print("\nStrategy Performance Summary")
    print(f"{'Strategy':<10} {'Mean Obj':>12} {'Std Dev':>10} {'Mean Spread':>14}")
    print("-" * 50)
    for s, stats in summary.items():
        print(f"{s:<10} {stats['mean_obj']:>12.4f} {stats['std_obj']:>10.4f} {stats['mean_spread']:>14.4f}")
