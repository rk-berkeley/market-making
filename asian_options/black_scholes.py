"""
black_scholes.py
----------------
Analytical Black-Scholes European call option pricing and Greeks.

Implements:
    - Black-Scholes call price (Theorem 1 in the report)
    - Numerical convexity verification via second finite differences
"""

import numpy as np
from scipy.stats import norm


def call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Compute the Black-Scholes European call price.

    Parameters
    ----------
    S0    : float  Initial asset price (S_0 > 0)
    K     : float  Strike price (K > 0)
    r     : float  Continuously-compounded risk-free rate
    sigma : float  Volatility (sigma > 0)
    T     : float  Time to maturity in years (T > 0)

    Returns
    -------
    float
        No-arbitrage call price C_BS(S0, K, r, sigma, T).
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def d1_d2(S0: float, K: float, r: float, sigma: float, T: float):
    """Return the auxiliary d1 and d2 parameters."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def delta(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes delta: dC/dS."""
    d1, _ = d1_d2(S0, K, r, sigma, T)
    return norm.cdf(d1)


def gamma(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes gamma: d²C/dS²."""
    d1, _ = d1_d2(S0, K, r, sigma, T)
    return norm.pdf(d1) / (S0 * sigma * np.sqrt(T))


def vega(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes vega: dC/d(sigma)."""
    d1, _ = d1_d2(S0, K, r, sigma, T)
    return S0 * norm.pdf(d1) * np.sqrt(T)


def convexity_check(
    S0: float = 50.0,
    r: float = 0.05,
    sigma: float = 0.30,
    T: float = 1.0,
    K_min: float = 35.0,
    K_max: float = 55.0,
    n_points: int = 200,
) -> dict:
    """Numerically verify that K -> C_BS(K) is convex via second finite differences.

    Returns a dict with:
        K_grid          : array of strike values
        prices          : array of BS call prices
        second_diffs    : array of second finite differences (all > 0 iff convex)
        min_second_diff : minimum second finite difference (> 0 confirms convexity)
    """
    K_grid = np.linspace(K_min, K_max, n_points)
    prices = np.array([call_price(S0, K, r, sigma, T) for K in K_grid])
    second_diffs = prices[:-2] - 2 * prices[1:-1] + prices[2:]
    return {
        "K_grid": K_grid,
        "prices": prices,
        "second_diffs": second_diffs,
        "min_second_diff": second_diffs.min(),
    }


if __name__ == "__main__":
    # Quick sanity check
    result = convexity_check()
    print(f"Min second finite difference: {result['min_second_diff']:.4e}  (must be > 0)")
    print(f"BS call at K=50: {call_price(50, 50, 0.05, 0.30, 1):.4f}")
