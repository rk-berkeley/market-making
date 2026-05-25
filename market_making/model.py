"""
model.py
--------
Core model components for optimal market-making under fad price dynamics.

Implements the stochastic model from:
    Barucci, Mathieu & Sánchez-Betancourt (2025),
    "Market Making with Fads, Informed, and Uninformed Traders."
    arXiv:2501.03658

Model state:
    S_t : mid-price (arithmetic Brownian motion with fad component)
    U_t : latent fad (Ornstein-Uhlenbeck mean-reverting process)
    Q_t : market maker's inventory (jump process)
    X_t : market maker's cash (jump process)

See report Section 3 for full mathematical derivation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ModelParams:
    """All parameters for the fad market-making model.

    Attributes
    ----------
    T         : Trading horizon
    S0        : Initial mid-price
    sigma     : Price volatility
    mu        : Price drift
    eta       : OU mean-reversion speed for the fad U_t
    q_weight  : Weight of fad in price BM decomposition  (denoted q in paper)
    phi       : Uninformed order arrival intensity baseline
    psi       : Informed order arrival intensity baseline
    k         : Price sensitivity of order arrivals
    gamma     : Fad sensitivity of informed order arrivals
    alpha     : Terminal inventory penalty
    varphi    : Running inventory penalty
    q_min     : Minimum inventory level (underline_q)
    q_max     : Maximum inventory level (bar_q)
    delta_inf : Lower bound on quote displacements
    S_minus   : Clipping level on ask side (S^-)
    S_plus    : Clipping level on bid side (S^+)
    N_steps   : Number of time discretisation steps
    M_paths   : Number of Monte Carlo paths
    seed      : Random seed
    """
    T:         float = 1.0
    S0:        float = 100.0
    sigma:     float = 1.0
    mu:        float = 0.0
    eta:       float = 10.0
    q_weight:  float = 0.6    # 'q' in the paper  (p = sqrt(1 - q^2))
    phi:       float = 15.0
    psi:       float = 15.0
    k:         float = 1.0
    gamma:     float = 1.0
    alpha:     float = 0.1
    varphi:    float = 0.1
    q_min:     int   = -10
    q_max:     int   = 10
    delta_inf: float = 5.0
    S_minus:   float = -np.inf
    S_plus:    float =  np.inf
    N_steps:   int   = 1000
    M_paths:   int   = 3000
    seed:      int   = 0

    @property
    def p_weight(self) -> float:
        """Fundamental noise weight (p = sqrt(1 - q^2))."""
        return np.sqrt(1.0 - self.q_weight ** 2)

    @property
    def dt(self) -> float:
        """Uniform time step size."""
        return self.T / self.N_steps

    @property
    def inventory_range(self) -> range:
        """All admissible inventory levels."""
        return range(self.q_min, self.q_max + 1)


def simulate_fad(params: ModelParams, rng: np.random.Generator) -> np.ndarray:
    """Simulate the latent fad process U_t using the Euler–Maruyama scheme.

    dU_t = -eta * U_t dt + dB_t,  U_0 = 0

    Parameters
    ----------
    params : ModelParams
    rng    : numpy random Generator (for reproducibility)

    Returns
    -------
    U : np.ndarray, shape (params.N_steps + 1,)
    """
    dt = params.dt
    dB = rng.standard_normal(params.N_steps) * np.sqrt(dt)
    U = np.zeros(params.N_steps + 1)
    for i in range(params.N_steps):
        U[i + 1] = U[i] - params.eta * U[i] * dt + dB[i]
    return U


def simulate_price(
    params: ModelParams, U: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Simulate the mid-price S_t given a pre-simulated fad path.

    dS_t = mu dt + sigma (p dZ_t + q dU_t)

    The price increment is constructed from independent Z increments and
    the already-simulated B increments embedded in U.

    Parameters
    ----------
    params : ModelParams
    U      : np.ndarray shape (N_steps + 1,) — fad path
    rng    : numpy random Generator

    Returns
    -------
    S : np.ndarray, shape (params.N_steps + 1,)
    """
    dt    = params.dt
    dZ    = rng.standard_normal(params.N_steps) * np.sqrt(dt)
    dU    = np.diff(U)          # dU_t = U_{t+1} - U_t
    S     = np.zeros(params.N_steps + 1)
    S[0]  = params.S0
    for i in range(params.N_steps):
        dS      = params.mu * dt + params.sigma * (params.p_weight * dZ[i] + params.q_weight * dU[i])
        S[i + 1] = S[i] + dS
    return S


def order_intensities(
    delta_a: float,
    delta_b: float,
    u: float,
    q: int,
    params: ModelParams,
) -> tuple[float, float]:
    """Compute the ask and bid order arrival intensities.

    lambda^a = (phi * exp(-k*delta^a) + psi * exp(-k*delta^a - gamma*(sigma*q*u v S^-)))
               * 1_{q > q_min}
    lambda^b = (phi * exp(-k*delta^b) + psi * exp(-k*delta^b + gamma*(sigma*q*u ^ S^+)))
               * 1_{q < q_max}

    Parameters
    ----------
    delta_a : Ask quote displacement
    delta_b : Bid quote displacement
    u       : Current fad value (or filtered estimate)
    q       : Current inventory level
    params  : ModelParams

    Returns
    -------
    (lambda_a, lambda_b) : tuple of floats
    """
    fad_a = np.maximum(params.sigma * params.q_weight * u, params.S_minus)
    fad_b = np.minimum(params.sigma * params.q_weight * u, params.S_plus)

    lam_a = (
        params.phi * np.exp(-params.k * delta_a)
        + params.psi * np.exp(-params.k * delta_a - params.gamma * fad_a)
    ) * (q > params.q_min)

    lam_b = (
        params.phi * np.exp(-params.k * delta_b)
        + params.psi * np.exp(-params.k * delta_b + params.gamma * fad_b)
    ) * (q < params.q_max)

    return lam_a, lam_b


def performance_objective(
    X_T: float, Q_T: int, S_T: float, inventory_path: np.ndarray,
    params: ModelParams,
) -> float:
    """Evaluate the realised performance criterion J for one path.

    J = X_T + Q_T * S_T - alpha * Q_T^2 - varphi * int_0^T Q_t^2 dt

    Parameters
    ----------
    X_T            : Terminal cash
    Q_T            : Terminal inventory
    S_T            : Terminal mid-price
    inventory_path : np.ndarray shape (N_steps + 1,) of Q_t values
    params         : ModelParams

    Returns
    -------
    float : Realised objective value
    """
    running_penalty = params.varphi * params.dt * np.sum(inventory_path[:-1] ** 2)
    return X_T + Q_T * S_T - params.alpha * Q_T ** 2 - running_penalty
