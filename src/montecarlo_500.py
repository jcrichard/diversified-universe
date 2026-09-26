"""
Monte Carlo comparison of universe-selection methods on 500 assets.

For each simulation:
  1. Draw a TRUE 500 x 500 correlation matrix with a random structure
     (factors with signed loadings, sectors, hedges, high and low correlations).
  2. Simulate T returns from it and compute the SAMPLE correlation matrix
     (what an investor actually sees).
  3. Each method selects p assets using the SAMPLE matrix.
  4. Each selection is scored on the TRUE matrix (real diversification)
     and on the SAMPLE matrix (in-sample).
"""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diversified_universe_fast import fast_min_universe, swap_search
from binary_selection import diversity_optimize, logdet_optimize, describe

N, T = 500, 750            # 500 assets, ~3 years of daily data
P_LIST = [10, 20, 50]
N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 else 100


# ---------------------------------------------------------------------------
# Random correlation structures
# ---------------------------------------------------------------------------
def random_true_corr(rng):
    """Mix of: global market factor, sectors, random signed factors, hedges."""
    n = N
    k = rng.integers(2, 31)                              # number of latent factors
    B = rng.standard_normal((n, k)) * rng.uniform(0.05, 0.6, size=k)

    # sectors / clusters with random strength (some tight, some loose)
    n_sect = rng.integers(3, 25)
    sect = rng.integers(0, n_sect, size=n)
    sect_load = rng.uniform(0.0, 1.5, size=n_sect)
    S = np.zeros((n, n_sect)); S[np.arange(n), sect] = sect_load[sect]

    # market factor on a random share of assets
    mkt_strength = rng.uniform(0.0, 2.0)                 # from no market to dominant market
    mkt = (rng.random(n) < rng.uniform(0.3, 1.0)) * mkt_strength * rng.uniform(0.5, 1.5, size=n)

    L = np.column_stack([B, S, mkt])
    # flip a random share of assets -> negative correlations (hedges, shorts, FX pairs)
    flip = np.where(rng.random(n) < rng.uniform(0.0, 0.3), -1.0, 1.0)
    L *= flip[:, None]

    idio = rng.uniform(0.3, 1.5, size=n) ** 2            # low to high idiosyncratic share
    Sigma = L @ L.T + np.diag(idio)
    d = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(d, d), L, np.sqrt(idio)


def sample_corr(L, idio_sd, rng):
    F = rng.standard_normal((T, L.shape[1]))
    R = F @ L.T + rng.standard_normal((T, N)) * idio_sd
    return np.corrcoef(R, rowvar=False)


# ---------------------------------------------------------------------------
# Methods (all select on the sample matrix)
# ---------------------------------------------------------------------------
def methods(Cs, p, rng):
    paper = fast_min_universe(Cs, p)
    return {
        "Paper (published)": paper,
        "Paper + swap": swap_search(Cs, paper, "min")[0],
        "Max diversity |rho|": diversity_optimize(Cs, p, "abs", n_starts=10)[0],
        "Max diversity rho": diversity_optimize(Cs, p, "rho", n_starts=10)[0],
        "Log-det": logdet_optimize(Cs, p, n_starts=5)[0],
        "Random": rng.choice(N, p, replace=False).tolist(),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    rows, t0 = [], time.time()
    for sim in range(N_SIM):
        Ct, L, idio = random_true_corr(rng)
        Cs = sample_corr(L, idio, rng)
        off = Ct[np.triu_indices(N, 1)]
        for p in P_LIST:
            timings = {}
            for name, S in methods(Cs, p, rng).items():
                true_m, samp_m = describe(Ct, S), describe(Cs, S)
                rows.append({
                    "sim": sim, "p": p, "method": name,
                    "univ_mean_rho": off.mean(), "univ_share_neg": (off < 0).mean(),
                    **{f"true_{k}": v for k, v in true_m.items()},
                    **{f"in_{k}": v for k, v in samp_m.items()},
                })
        if (sim + 1) % 10 == 0:
            print(f"{sim + 1}/{N_SIM} sims  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "montecarlo_500_results.csv"), index=False)
    print("saved", len(df), "rows")
