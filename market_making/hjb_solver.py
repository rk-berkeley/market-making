"""
hjb_solver.py
-------------
Solves the approximate HJB ODE system for the full-information (FI) optimal
market-making strategy.

The quadratic ansatz V(t, q, u) = q^2 * A(t) + q * B(t, u) + C(t, u)
reduces the HJB PDE to:

    1) A scalar Riccati ODE for A(t)  --> closed-form solution
    2) A linear ODE for B(t, u) = b0(t) + u * b1(t)
    3) A linear ODE for C(t, u) = c0(t) + u * c1(t) + u^2 * c2(t)

All ODEs run backward from terminal conditions A(T)=-alpha, B(T,.)=C(T,.)=0.

Reference: Barucci, Mathieu & Sánchez-Betancourt (2025), arXiv:2501.03658.
"""

import numpy as np
from scipy.integrate import solve_ivp
from market_making.model import ModelParams


# ---------------------------------------------------------------------------
# A(t): Riccati ODE
# ---------------------------------------------------------------------------

def _riccati_rhs(t: float, A: list, params: ModelParams) -> list:
    """RHS of the Riccati ODE for A(t): dA/dt = varphi - kappa * A^2."""
    kappa = 4 * (params.phi + params.psi) * np.exp(-1) * params.k
    return [params.varphi - kappa * A[0] ** 2]


def solve_A(params: ModelParams, t_grid: np.ndarray) -> np.ndarray:
    """Solve the Riccati ODE for A(t) on t_grid using the closed-form solution.

    A(t) = sqrt(varphi/kappa) * (1 - exp(2*sqrt(varphi*kappa)*(T-t)) * beta)
                                / (1 + exp(2*sqrt(varphi*kappa)*(T-t)) * beta)
    where  beta = (sqrt(varphi) + sqrt(kappa)*alpha) / (sqrt(varphi) - sqrt(kappa)*alpha)

    Parameters
    ----------
    params : ModelParams
    t_grid : 1-D array of time points in [0, T]

    Returns
    -------
    A_vals : np.ndarray, same shape as t_grid
    """
    kappa = 4.0 * (params.phi + params.psi) * np.exp(-1.0) * params.k
    sv = np.sqrt(params.varphi)
    sk = np.sqrt(kappa)
    beta = (sv + sk * params.alpha) / (sv - sk * params.alpha)
    tau  = params.T - t_grid                            # time-to-maturity
    exp_term = np.exp(2.0 * np.sqrt(params.varphi * kappa) * tau)
    A_vals   = np.sqrt(params.varphi / kappa) * (1.0 - exp_term * beta) / (1.0 + exp_term * beta)
    return A_vals


# ---------------------------------------------------------------------------
# B(t, u) = b0(t) + u * b1(t): Linear ODE system
# ---------------------------------------------------------------------------

def _b_rhs(t: float, y: list, A_interp, params: ModelParams) -> list:
    """RHS of the linear system [db0/dt, db1/dt]."""
    b0, b1 = y
    A  = float(A_interp(t))
    kappa  = 4.0 * (params.phi + params.psi) * np.exp(-1.0) * params.k
    lam    = kappa * A                                  # 4(phi+psi) e^{-1} k A

    db0 = -params.mu + lam * b0
    db1 = (params.eta - lam) * b1 \
          - 4.0 * np.exp(-1.0) * params.psi * params.q_weight * params.sigma * params.gamma * A \
          - 4.0 * np.exp(-1.0) * params.k * params.q_weight * params.gamma * params.sigma * params.psi * A ** 2
    return [db0, db1]


def solve_B(params: ModelParams, t_grid: np.ndarray, A_vals: np.ndarray):
    """Solve the linear ODE system for b0(t) and b1(t).

    Returns
    -------
    b0_vals, b1_vals : np.ndarray, same shape as t_grid
    """
    from scipy.interpolate import interp1d
    A_interp = interp1d(t_grid, A_vals, kind="cubic", fill_value="extrapolate")

    # Integrate backward: run from T to 0, then reverse
    sol = solve_ivp(
        fun=lambda t, y: _b_rhs(t, y, A_interp, params),
        t_span=(params.T, 0.0),
        y0=[0.0, 0.0],           # terminal conditions B(T,.) = 0
        t_eval=t_grid[::-1],
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    b0_vals = sol.y[0, ::-1]
    b1_vals = sol.y[1, ::-1]
    return b0_vals, b1_vals


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def solve_hjb_coefficients(params: ModelParams):
    """Compute all ODE coefficients on a fine uniform grid.

    Returns
    -------
    t_grid : np.ndarray shape (N_steps+1,)
    A      : np.ndarray shape (N_steps+1,)   -- Riccati solution
    b0     : np.ndarray shape (N_steps+1,)   -- constant part of B
    b1     : np.ndarray shape (N_steps+1,)   -- linear-in-u part of B
    """
    t_grid = np.linspace(0.0, params.T, params.N_steps + 1)
    A      = solve_A(params, t_grid)
    b0, b1 = solve_B(params, t_grid, A)
    return t_grid, A, b0, b1


def optimal_displacements(
    t_idx: int,
    q: int,
    u: float,
    A: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    params: ModelParams,
) -> tuple[float, float]:
    """Compute the FI optimal ask and bid displacements.

    delta^{a,*} = 1/k + (2q - 1) A(t) + B(t, u)
    delta^{b,*} = 1/k - (2q + 1) A(t) - B(t, u)

    Clipped at -delta_inf from below.
    """
    B = b0[t_idx] + u * b1[t_idx]
    d_a = 1.0 / params.k + (2 * q - 1) * A[t_idx] + B
    d_b = 1.0 / params.k - (2 * q + 1) * A[t_idx] - B
    d_a = max(d_a, -params.delta_inf)
    d_b = max(d_b, -params.delta_inf)
    return d_a, d_b
