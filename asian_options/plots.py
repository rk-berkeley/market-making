"""
plots.py
--------
All figure generation for the Asian Options section (Section 2 of the report).

Reproduces:
    fig_q1_convexity.pdf    -- BS call price convexity in K  (Q1)
    fig_q3_ci_length.pdf    -- 95% CI width comparison       (Q3)
    fig_q4_comparison.pdf   -- Asian vs. European prices     (Q4)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from asian_options.black_scholes import call_price, convexity_check
from asian_options.monte_carlo import asian_option_mc, confidence_interval_95

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
})

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"


def plot_q1_convexity(
    S0: float = 50.0,
    r: float = 0.05,
    sigma: float = 0.30,
    T: float = 1.0,
    save_path: str = "fig_q1_convexity.pdf",
) -> None:
    """Plot BS call price as a function of strike K, confirming convexity."""
    res = convexity_check(S0, r, sigma, T)
    K_grid = res["K_grid"]
    prices = res["prices"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_grid, prices, color=BLUE, linewidth=2.2)
    ax.set_xlabel("Strike $K$", fontsize=13)
    ax.set_ylabel("$C_{\\mathrm{BS}}(K)$", fontsize=13)
    ax.set_title(
        f"Black-Scholes call price vs. strike\n"
        f"$S_0={S0}$, $r={r}$, $\\sigma={sigma}$, $T={T}$",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_q3_ci_length(
    S0: float = 50.0,
    r: float = 0.05,
    sigma: float = 0.30,
    T: float = 1.0,
    M: int = 200_000,
    K_list: list = None,
    save_path: str = "fig_q3_ci_length.pdf",
) -> None:
    """Bar chart comparing 95% CI widths: arithmetic vs. geometric Asian calls."""
    if K_list is None:
        K_list = [35, 40, 45, 50, 55]

    widths_a, widths_g = [], []
    for K in K_list:
        _, sa = asian_option_mc(S0, K, r, sigma, T, method="arithmetic")
        _, sg = asian_option_mc(S0, K, r, sigma, T, method="geometric")
        widths_a.append(2 * 1.96 * sa / np.sqrt(M))
        widths_g.append(2 * 1.96 * sg / np.sqrt(M))

    x, w = np.arange(len(K_list)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, widths_a, width=w, label="Arithmetic", color=BLUE,   alpha=0.85)
    ax.bar(x + w / 2, widths_g, width=w, label="Geometric",  color=ORANGE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(K_list)
    ax.set_xlabel("Strike $K$", fontsize=13)
    ax.set_ylabel("95% CI width", fontsize=13)
    ax.set_title("Confidence interval widths: arithmetic vs. geometric averaging", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_q4_comparison(
    S0: float = 50.0,
    r: float = 0.05,
    sigma: float = 0.30,
    T: float = 1.0,
    K_list: list = None,
    save_path: str = "fig_q4_comparison.pdf",
) -> None:
    """Line plot verifying the price ordering: Geom <= Arith <= BS European."""
    if K_list is None:
        K_list = [35, 40, 45, 50, 55]

    pa_list, pg_list, pbs_list = [], [], []
    for K in K_list:
        pa, _ = asian_option_mc(S0, K, r, sigma, T, method="arithmetic")
        pg, _ = asian_option_mc(S0, K, r, sigma, T, method="geometric")
        pa_list.append(pa)
        pg_list.append(pg)
        pbs_list.append(call_price(S0, K, r, sigma, T))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_list, pbs_list, "o-",  color=BLUE,   label="European $C_{\\mathrm{BS}}$", linewidth=2.2)
    ax.plot(K_list, pa_list,  "s--", color=ORANGE,  label="Asian (arithmetic)",          linewidth=2.2)
    ax.plot(K_list, pg_list,  "^:",  color=GREEN,   label="Asian (geometric)",            linewidth=2.2)
    ax.set_xlabel("Strike $K$", fontsize=13)
    ax.set_ylabel("Option price", fontsize=13)
    ax.set_title(
        f"Price ordering: Geometric $\\leq$ Arithmetic $\\leq$ European\n"
        f"$S_0={S0}$, $r={r}$, $\\sigma={sigma}$, $T={T}$",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_q1_convexity()
    plot_q3_ci_length()
    plot_q4_comparison()
