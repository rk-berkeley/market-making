"""
kalman_filter.py
----------------
Kalman-Bucy filter for estimating the latent fad U_t from the observed price.

In the partial-information setting the market maker only sees the price
filtration F^S. The fad U_t is hidden. The conditional mean and variance

    Û_t := E[U_t | F^S_t],   P_t := E[(U_t - Û_t)^2 | F^S_t]

satisfy linear SDEs whose diffusion coefficient comes from the
Riccati ODE for P_t (Kalman-Bucy, Theorem 4 in the report).

Reference: Bain & Crisan (2009), "Fundamentals of Stochastic Filtering."
"""

import numpy as np
from market_making.model import ModelParams


def kalman_riccati(params: ModelParams) -> np.ndarray:
    """Solve the Riccati ODE for the conditional variance P_t.

    dP/dt = -(eta * q)^2 * P^2 - P * (2*eta - 2*eta*q^2) + p^2
    P(0)  = 0

    This ODE is deterministic, so we solve it once on the full time grid.

    Returns
    -------
    P_vals : np.ndarray shape (N_steps + 1,)
    """
    q  = params.q_weight
    p  = params.p_weight
    dt = params.dt
    P  = np.zeros(params.N_steps + 1)
    P[0] = 0.0
    for i in range(params.N_steps):
        dP    = (-(params.eta * q) ** 2 * P[i] ** 2
                 - P[i] * (2 * params.eta - 2 * params.eta * q ** 2)
                 + p ** 2) * dt
        P[i + 1] = P[i] + dP
    return P


def kalman_filter_path(
    S_path: np.ndarray,
    P_vals: np.ndarray,
    params: ModelParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run the Kalman-Bucy filter along one realised price path.

    dÛ_t = -eta * Û_t dt + sigma^{-1} * (-eta * q * P_t + q) * dI_t
    dI_t  = dS_t - (mu - eta * sigma * q * Û_t) dt     (innovation)

    Parameters
    ----------
    S_path  : np.ndarray shape (N_steps + 1,)  -- realised mid-price path
    P_vals  : np.ndarray shape (N_steps + 1,)  -- Riccati solution
    params  : ModelParams
    rng     : numpy random Generator (unused here — filter is deterministic
              given S_path, included for API consistency)

    Returns
    -------
    U_hat : np.ndarray shape (N_steps + 1,)  -- filtered fad estimate Û_t
    """
    dt     = params.dt
    q      = params.q_weight
    eta    = params.eta
    sigma  = params.sigma
    mu     = params.mu

    U_hat      = np.zeros(params.N_steps + 1)
    U_hat[0]   = 0.0
    dS         = np.diff(S_path)                # observed price increments

    for i in range(params.N_steps):
        gain    = (q - eta * q * P_vals[i]) / sigma
        innov   = dS[i] - (mu - eta * sigma * q * U_hat[i]) * dt
        dU_hat  = -eta * U_hat[i] * dt + gain * innov
        U_hat[i + 1] = U_hat[i] + dU_hat

    return U_hat
