"""
monte_carlo.py
--------------
Monte Carlo pricing engine for arithmetic and geometric Asian call options.

Implements:
    - Exact log-normal GBM path simulation (no time-discretisation error)
    - Arithmetic and geometric average payoff computation
    - 95% confidence interval construction (CLT-based)

References
----------
Kemna & Vorst (1990), "A pricing method for options based on average asset values."
"""

import numpy as np
from typing import Literal, Tuple


def simulate_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    M: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate M risk-neutral GBM paths on a uniform grid of n_steps+1 dates.

    Uses the exact log-normal increment to avoid any Euler-Maruyama bias:
        S_{t+dt} = S_t * exp((r - sigma²/2)*dt + sigma*sqrt(dt)*Z)

    Parameters
    ----------
    S0      : Initial asset price
    r       : Risk-free rate
    sigma   : Volatility
    T       : Maturity (years)
    n_steps : Number of time steps  (n in the report)
    M       : Number of Monte Carlo paths
    seed    : NumPy random seed for reproducibility

    Returns
    -------
    S : np.ndarray, shape (M, n_steps + 1)
        Simulated asset price paths.
    """
    np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.randn(M, n_steps)
    S = np.empty((M, n_steps + 1))
    S[:, 0] = S0
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )
    return S


def asian_option_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 50,
    M: int = 200_000,
    method: Literal["arithmetic", "geometric"] = "arithmetic",
    seed: int = 42,
) -> Tuple[float, float]:
    """Price an Asian call option via Monte Carlo simulation.

    The discounted payoff is  exp(-rT) * max(A_T - K, 0), where A_T is the
    arithmetic or geometric average of the simulated asset price path.

    Parameters
    ----------
    S0      : Initial asset price
    K       : Strike price
    r       : Risk-free rate
    sigma   : Volatility
    T       : Maturity
    n_steps : Number of discretisation steps (n)
    M       : Monte Carlo sample size
    method  : "arithmetic" (Eq. 3 in report) or "geometric" (Eq. 4)
    seed    : Random seed

    Returns
    -------
    price : float   Discounted MC estimate of the Asian call price
    std   : float   Sample standard deviation of discounted payoffs
    """
    S = simulate_paths(S0, r, sigma, T, n_steps, M, seed=seed)

    if method == "arithmetic":
        A = np.mean(S, axis=1)
    elif method == "geometric":
        A = np.exp(np.mean(np.log(S), axis=1))
    else:
        raise ValueError(f"method must be 'arithmetic' or 'geometric', got {method!r}")

    disc_payoff = np.exp(-r * T) * np.maximum(A - K, 0.0)
    return disc_payoff.mean(), disc_payoff.std(ddof=1)


def confidence_interval_95(
    mean: float, std: float, M: int
) -> Tuple[float, float, float]:
    """Construct the asymptotic 95% Monte Carlo confidence interval.

    Uses the CLT: (X_bar - mu) / (s / sqrt(M)) -> N(0,1).

    Parameters
    ----------
    mean : Sample mean
    std  : Sample standard deviation
    M    : Sample size

    Returns
    -------
    lower : float  Lower endpoint of CI
    upper : float  Upper endpoint of CI
    width : float  Full width of CI  (= 2 * 1.96 * std / sqrt(M))
    """
    hw = 1.96 * std / np.sqrt(M)
    return mean - hw, mean + hw, 2 * hw


def full_parameter_sweep(
    S0: float = 50.0,
    n_steps: int = 50,
    M: int = 200_000,
    seed: int = 42,
) -> list[dict]:
    """Run the full parameter sweep from Tables 3 & 4 in the report.

    Parameter grid:
        r      : 0.00 (sigma in {0.2, 0.3, 0.4}), 0.03, 0.04, 0.05, 1.00
        K      : {35, 40, 45, 50, 55}
        T      : {1, 2}

    Returns
    -------
    rows : list of dicts, each containing r, sigma, T, K, Arith price,
           Arith CI width, Geom price, Geom CI width, BS price.
    """
    from asian_options.black_scholes import call_price

    param_rows = [
        (0.00, [0.2, 0.3, 0.4]),
        (0.03, [0.2]),
        (0.04, [0.4]),
        (0.05, [0.3]),
        (1.00, [0.1]),
    ]
    K_list = [35, 40, 45, 50, 55]
    T_list = [1, 2]

    rows = []
    for r, sigma_list in param_rows:
        for sigma in sigma_list:
            for T in T_list:
                for K in K_list:
                    pa, sa = asian_option_mc(S0, K, r, sigma, T, n_steps, M, "arithmetic", seed)
                    pg, sg = asian_option_mc(S0, K, r, sigma, T, n_steps, M, "geometric",  seed)
                    _, _, len_a = confidence_interval_95(pa, sa, M)
                    _, _, len_g = confidence_interval_95(pg, sg, M)
                    rows.append(dict(
                        r=r, sigma=sigma, T=T, K=K,
                        Arith=pa,       CI_A_len=len_a,
                        Geom=pg,        CI_G_len=len_g,
                        BS=call_price(S0, K, r, sigma, T),
                    ))
    return rows


if __name__ == "__main__":
    # Example: single pricing call
    price_a, std_a = asian_option_mc(50, 50, 0.05, 0.30, 1, method="arithmetic")
    price_g, std_g = asian_option_mc(50, 50, 0.05, 0.30, 1, method="geometric")
    _, _, w_a = confidence_interval_95(price_a, std_a, 200_000)
    _, _, w_g = confidence_interval_95(price_g, std_g, 200_000)

    print(f"Arithmetic Asian call: {price_a:.4f}  (95% CI width: {w_a:.4f})")
    print(f"Geometric  Asian call: {price_g:.4f}  (95% CI width: {w_g:.4f})")
    print(f"CI width reduction:    {(w_a - w_g) / w_a * 100:.2f}%")
