"""
Pure in/out universe selection (no weights) for two criteria:

  A. Maximum diversity : minimise the sum of pairwise correlations of the
                         selected assets  (rho, |rho| or rho^2)
  B. Log-determinant   : maximise log det C_S  (the 'volume' of the universe)

For each criterion:
  1. greedy construction       -> good starting point, very fast
  2. swap local search         -> exchange 1 asset in / 1 asset out until no
                                  exchange improves (exact, vectorised gains)
  3. multi-start               -> repeat 2 from random starts, keep the best
  4. exact solver (small n)    -> to measure how close 1-3 get to the optimum
"""

from itertools import combinations

import numpy as np


# ===========================================================================
# A. MAXIMUM DIVERSITY:  min sum_{i<j in S} w_ij,   w = rho, |rho| or rho^2
# ===========================================================================
def _pair_matrix(C, kind="abs"):
    W = {"rho": C, "abs": np.abs(C), "sq": C ** 2}[kind].astype(float).copy()
    np.fill_diagonal(W, 0.0)
    return W


def diversity_value(W, S):
    S = list(S)
    return W[np.ix_(S, S)].sum() / 2.0


def diversity_greedy(W, p):
    """Start from the lowest pair, then add the asset with the smallest total
    correlation to the assets already selected. O(n p)."""
    n = W.shape[0]
    A = W + np.diag(np.full(n, np.inf))
    i, j = np.unravel_index(np.argmin(A), A.shape)
    S = [int(i), int(j)]
    s = W[:, i] + W[:, j]                  # s_k = sum_{m in S} w_km
    free = np.ones(n, bool); free[S] = False
    while len(S) < p:
        k = int(np.flatnonzero(free)[np.argmin(s[free])])
        S.append(k); free[k] = False
        s += W[:, k]
    return S


def diversity_swap(W, S, max_iter=10_000):
    """Best-improvement 1-swap search.
    Gain of removing i (in S) and adding j (not in S):
        delta = s_j - s_i - w_ij        (negative = improvement)
    where s_k = sum_{m in S} w_km. All p x (n-p) moves evaluated at once."""
    n = W.shape[0]
    S = np.array(S)
    inS = np.zeros(n, bool); inS[S] = True
    s = W[:, S].sum(axis=1)
    for _ in range(max_iter):
        out = np.flatnonzero(~inS)
        D = s[out][None, :] - s[S][:, None] - W[np.ix_(S, out)]
        a, b = np.unravel_index(np.argmin(D), D.shape)
        if D[a, b] >= -1e-12:
            break
        i, j = S[a], out[b]
        S[a] = j; inS[i] = False; inS[j] = True
        s += W[:, j] - W[:, i]
    return S.tolist()


def diversity_optimize(C, p, kind="abs", n_starts=20, seed=0):
    """Greedy + swap, then random multi-start + swap. Returns (S, value)."""
    W = _pair_matrix(C, kind)
    rng = np.random.default_rng(seed)
    best = diversity_swap(W, diversity_greedy(W, p))
    best_v = diversity_value(W, best)
    for _ in range(n_starts):
        S = diversity_swap(W, rng.choice(W.shape[0], p, replace=False))
        v = diversity_value(W, S)
        if v < best_v - 1e-12:
            best, best_v = S, v
    return sorted(best), best_v


def diversity_exact_milp(C, p, kind="abs", time_limit=120):
    """Exact solution by linearised binary program (scipy/HiGHS).
    Variables x_i (asset selected) and y_ij = x_i x_j.
    Practical up to roughly n = 40-60; beyond that use a commercial solver
    (Gurobi / CPLEX handle the binary quadratic form directly)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix

    W = _pair_matrix(C, kind)
    n = W.shape[0]
    pairs = list(combinations(range(n), 2))
    m = len(pairs)
    c = np.concatenate([np.zeros(n), [W[i, j] for i, j in pairs]])

    rows, lb, ub = [], [], []
    A = lil_matrix((1 + 3 * m, n + m))
    A[0, :n] = 1; lb.append(p); ub.append(p)
    r = 1
    for k, (i, j) in enumerate(pairs):
        y = n + k
        if W[i, j] >= 0:            # minimisation pushes y down: need y >= x_i + x_j - 1
            A[r, i] = 1; A[r, j] = 1; A[r, y] = -1; lb.append(-np.inf); ub.append(1); r += 1
        else:                       # pushes y up: need y <= x_i, y <= x_j
            A[r, y] = 1; A[r, i] = -1; lb.append(-np.inf); ub.append(0); r += 1
            A[r, y] = 1; A[r, j] = -1; lb.append(-np.inf); ub.append(0); r += 1
    A = A[:r].tocsr()
    res = milp(c, constraints=LinearConstraint(A, lb, ub),
               integrality=np.r_[np.ones(n), np.zeros(m)],
               bounds=Bounds(0, 1), options={"time_limit": time_limit})
    S = np.flatnonzero(res.x[:n] > 0.5).tolist()
    return S, diversity_value(W, S)


# ===========================================================================
# B. LOG-DETERMINANT:  max log det C_S
# ===========================================================================
def logdet_value(C, S):
    S = list(S)
    return np.linalg.slogdet(C[np.ix_(S, S)])[1]


def logdet_greedy(C, p):
    """Pivoted Cholesky. d_j = residual variance of asset j after regressing it
    on the selected assets = 1 - R^2_{j|S}. Each step adds argmax d_j, which
    increases log det by log d_j. O(n p^2) overall."""
    n = C.shape[0]
    d = np.diag(C).astype(float).copy()
    L = np.zeros((p, n))
    S = []
    for t in range(p):
        d_masked = d.copy(); d_masked[S] = -np.inf
        k = int(np.argmax(d_masked))
        S.append(k)
        e = (C[k] - L[:t, k] @ L[:t]) / np.sqrt(d[k])
        L[t] = e
        d = d - e ** 2
    return S


def logdet_swap(C, S, max_iter=10_000):
    """Best-improvement 1-swap search, all moves scored exactly at once.
    With M = C_S^{-1}, q_j = C[S, j], P = M Q:
        log det(S - i + j) - log det(S) = log M_ii + log( 1 - q_j'M q_j + P_ij^2 / M_ii )
    """
    n = C.shape[0]
    S = list(S)
    for _ in range(max_iter):
        M = np.linalg.inv(C[np.ix_(S, S)])
        out = np.setdiff1d(np.arange(n), S)
        Q = C[np.ix_(S, out)]
        P = M @ Q
        base = np.diag(C)[out] - (Q * P).sum(axis=0)
        mii = np.diag(M)
        R = base[None, :] + P ** 2 / mii[:, None]
        G = np.log(mii)[:, None] + np.log(np.maximum(R, 1e-300))
        a, b = np.unravel_index(np.argmax(G), G.shape)
        if G[a, b] <= 1e-10:
            break
        S[a] = int(out[b])
    return S


def logdet_optimize(C, p, n_starts=10, seed=0):
    rng = np.random.default_rng(seed)
    best = logdet_swap(C, logdet_greedy(C, p))
    best_v = logdet_value(C, best)
    for _ in range(n_starts):
        S = logdet_swap(C, rng.choice(C.shape[0], p, replace=False).tolist())
        v = logdet_value(C, S)
        if v > best_v + 1e-10:
            best, best_v = S, v
    return sorted(best), best_v


# ===========================================================================
# Diagnostics shared by all methods
# ===========================================================================
def describe(C, S):
    S = list(S)
    sub = C[np.ix_(S, S)]
    lam = np.linalg.eigvalsh(sub)[::-1]
    off = sub[np.triu_indices(len(S), 1)]
    return {
        "mean_rho": off.mean(),
        "mean_abs_rho": np.abs(off).mean(),
        "lambda0": lam[0],
        "pi0": lam[0] / lam.sum(),
        "eff_factors": lam.sum() ** 2 / (lam ** 2).sum(),   # participation ratio
        "logdet": np.linalg.slogdet(sub)[1],
    }


# ===========================================================================
# Demo
# ===========================================================================
if __name__ == "__main__":
    import time

    def synthetic(n, T, k, seed):
        rng = np.random.default_rng(seed)
        B = rng.standard_normal((n, k)) * rng.uniform(0.0, 0.8, size=(1, k))
        B[: n // 3] += 0.7 * rng.standard_normal()
        R = rng.standard_normal((T, k)) @ B.T + rng.standard_normal((T, n))
        return np.corrcoef(R, rowvar=False)

    # ---- 1. accuracy vs exact optimum on small universes -------------------
    print("Accuracy check (n = 30, p = 8)")
    C = synthetic(30, 1000, 6, seed=3)
    p = 8
    t = time.time(); S_ex, v_ex = diversity_exact_milp(C, p); t_ex = time.time() - t
    S_h, v_h = diversity_optimize(C, p)
    print(f"  max diversity  exact {v_ex:.4f} ({t_ex:.1f}s)   heuristic {v_h:.4f}")

    t = time.time()
    v_bf = max(logdet_value(C, s) for s in combinations(range(30), p))
    t_bf = time.time() - t
    print(f"  log-det        exact {v_bf:.4f} ({t_bf:.1f}s, 5.9M subsets)   "
          f"greedy {logdet_value(C, logdet_greedy(C, p)):.4f}   "
          f"heuristic {logdet_optimize(C, p)[1]:.4f}")

    # ---- 2. scale: 1000 assets ---------------------------------------------
    print("\n1000 assets, p = 20")
    C = synthetic(1000, 2000, 25, seed=1)
    p = 20
    t = time.time(); S_div, _ = diversity_optimize(C, p); t_div = time.time() - t
    t = time.time(); S_ld, _ = logdet_optimize(C, p); t_ld = time.time() - t
    try:
        from diversified_universe_fast import fast_min_universe, swap_search
        S_eig, _ = swap_search(C, fast_min_universe(C, p), "min")
        rows = [("Max diversity", S_div, t_div), ("Log-det", S_ld, t_ld), ("Paper min eig", S_eig, None)]
    except ImportError:
        rows = [("Max diversity", S_div, t_div), ("Log-det", S_ld, t_ld)]

    print(f"  {'method':<15}{'time':>7}{'mean|rho|':>11}{'lambda0':>9}{'pi0':>8}{'eff.fact':>10}{'logdet':>9}")
    for name, S, tt in rows:
        d = describe(C, S)
        ts = f"{tt:.2f}s" if tt is not None else "  -"
        print(f"  {name:<15}{ts:>7}{d['mean_abs_rho']:>11.4f}{d['lambda0']:>9.3f}"
              f"{d['pi0']:>8.1%}{d['eff_factors']:>10.2f}{d['logdet']:>9.3f}")
