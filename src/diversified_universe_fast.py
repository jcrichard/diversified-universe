"""
Scalable min / max diversified universe selection for large n (e.g. 1000 assets).

Why a separate version:
  * the naive forward step recomputes n eigendecompositions per added asset;
  * the naive backward step starts from an n x n matrix and needs ~n^2 full
    eigendecompositions of size up to n  ->  O(n^5), infeasible at n = 1000.

What this file adds:
  1. fast_min_universe : EXACT greedy forward step, but each candidate is scored
     through the secular equation of the bordered matrix, so one p x p
     eigendecomposition per step serves all n candidates.
  2. fast_max_universe : backward elimination scored by a Rayleigh-quotient
     bound (drop the asset with the weakest loading on the first eigenvector),
     with warm-started Lanczos for the top eigenpair. ~O(n^3) overall.
  3. swap_search       : 1-for-1 exchange local search to polish either result.
  4. bounds            : certificates telling you how far from optimal you can be
     without exhaustive search.
"""

import numpy as np
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# 1. Fast exact forward greedy (min first eigenvalue)
# ---------------------------------------------------------------------------
def _largest_root_bordered(lam, Z, d, iters=80):
    """
    Largest eigenvalue of [[C_S, c_j], [c_j', d_j]] for many candidates j at once.
    lam : (m,) eigenvalues of C_S
    Z   : (m, k) projections U' c_j for k candidates
    d   : (k,) diagonal entries C_jj
    Solves x - d - sum_i z_i^2 / (x - lam_i) = 0 by vectorised bisection on
    (max(lam_max, d), max(lam_max, d) + ||z||].
    """
    lam_max = lam.max()
    base = np.maximum(lam_max, d)
    lo = base + 1e-14
    hi = base + np.sqrt((Z ** 2).sum(axis=0)) + 1e-9
    Z2 = Z ** 2
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = mid - d - (Z2 / (mid[None, :] - lam[:, None])).sum(axis=0)
        pos = f > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return hi


def fast_min_universe(C, p, start=None):
    """
    Same result as the paper's forward algorithm (up to ties), much faster.
    start : optional initial list of assets (default: least correlated pair).
    Cost per step: one p x p eigendecomposition + O(n p^2).
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    if start is None:
        A = np.abs(C).copy()
        np.fill_diagonal(A, np.inf)
        i, j = np.unravel_index(np.argmin(A), A.shape)
        selected = [int(i), int(j)]
    else:
        selected = list(start)
    mask = np.ones(n, dtype=bool)
    mask[selected] = False

    while len(selected) < p:
        lam, U = np.linalg.eigh(C[np.ix_(selected, selected)])
        cand = np.flatnonzero(mask)
        Z = U.T @ C[np.ix_(selected, cand)]          # (m, k)
        new_lam = _largest_root_bordered(lam, Z, np.diag(C)[cand])
        best = cand[np.argmin(new_lam)]
        selected.append(int(best))
        mask[best] = False
    return selected


# ---------------------------------------------------------------------------
# 2. Fast backward elimination (max first eigenvalue)
# ---------------------------------------------------------------------------
def _top_eig(M, v0=None):
    if M.shape[0] <= 60:
        w, V = np.linalg.eigh(M)
        return w[-1], V[:, -1]
    w, V = eigsh(M, k=1, which="LA", v0=v0, tol=1e-10)
    return w[0], V[:, 0]


def fast_max_universe(C, p, batch=1):
    """
    Backward elimination. For each asset k, the first eigenvector u of the
    current universe restricted to the other assets gives a lower bound on
    the first eigenvalue after deleting k:
        Lambda(-k) >= (Lambda - 2 Lambda u_k^2 + C_kk u_k^2) / (1 - u_k^2)
    We delete the asset with the largest bound (for a correlation matrix this
    is simply the smallest |u_k|). batch > 1 deletes several per iteration to
    go even faster while the universe is still large.
    """
    C = np.asarray(C, dtype=float)
    selected = np.arange(C.shape[0])
    u = None
    while len(selected) > p:
        M = C[np.ix_(selected, selected)]
        lam, u = _top_eig(M, v0=u)
        u2 = np.minimum(u ** 2, 1 - 1e-12)
        score = (lam - 2 * lam * u2 + np.diag(M) * u2) / (1 - u2)
        k = min(batch, len(selected) - p, max(1, (len(selected) - p) // 10))
        drop = np.argsort(-score)[:k]
        keep = np.setdiff1d(np.arange(len(selected)), drop)
        selected, u = selected[keep], u[keep]
    return sorted(selected.tolist())


# ---------------------------------------------------------------------------
# 3. Swap local search (polishes either greedy output)
# ---------------------------------------------------------------------------
def swap_search(C, selected, mode="min", max_passes=20):
    """
    Try every (in, out) exchange; accept the best improving one; repeat until
    no single swap improves. A 1-swap local optimum is a much stronger result
    than a greedy path, and it's what you should report for n = 1000.
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    sel = list(selected)
    sign = 1.0 if mode == "min" else -1.0
    cur = sign * np.linalg.eigvalsh(C[np.ix_(sel, sel)])[-1]

    for _ in range(max_passes):
        out_set = np.setdiff1d(np.arange(n), sel)
        best_val, best_move = cur, None
        for pos in range(len(sel)):
            rest = sel[:pos] + sel[pos + 1:]
            lam, U = np.linalg.eigh(C[np.ix_(rest, rest)])
            Z = U.T @ C[np.ix_(rest, out_set)]
            vals = sign * _largest_root_bordered(lam, Z, np.diag(C)[out_set])
            j = np.argmin(vals)
            if vals[j] < best_val - 1e-10:
                best_val, best_move = vals[j], (pos, int(out_set[j]))
        if best_move is None:
            break
        pos, new = best_move
        sel[pos] = new
        cur = best_val
    return sel, sign * cur


# ---------------------------------------------------------------------------
# 4. Optimality certificates (no enumeration needed)
# ---------------------------------------------------------------------------
def bounds(C, p):
    """
    For a CORRELATION matrix:
      min problem : any p-subset has Lambda_0 >= 1           (trace argument)
      max problem : any p-subset has Lambda_0 <= min(
                        p,
                        Lambda_0(full matrix),               (Cauchy interlacing)
                        max_i [1 + sum of the p-1 largest |rho_ij|, j != i]  (Gershgorin/Morrison)
                    )
    If your greedy value is close to these, you are provably close to optimal.
    """
    C = np.asarray(C, dtype=float)
    A = np.abs(C).copy()
    np.fill_diagonal(A, 0.0)
    topk = -np.sort(-A, axis=1)[:, : p - 1].sum(axis=1)
    gersh = 1.0 + topk.max()
    full = _top_eig(C)[0]
    return {"min_lower": 1.0, "max_upper": min(p, full, gersh)}


# ---------------------------------------------------------------------------
# Demo: 1000 assets
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from diversified_universe import min_universe, max_universe

    rng = np.random.default_rng(1)
    n, T, k = 1000, 2000, 25
    B = rng.standard_normal((n, k)) * rng.uniform(0.0, 0.8, size=(1, k))
    B[: n // 3] += 0.7 * rng.standard_normal((1, 1))            # a big common block
    R = rng.standard_normal((T, k)) @ B.T + rng.standard_normal((T, n))
    C = np.corrcoef(R, rowvar=False)

    p = 20
    b = bounds(C, p)
    print(f"n = {n}, p = {p}")

    # --- MIN ----------------------------------------------------------------
    t = time.time(); s_naive = min_universe(C, p); t_naive = time.time() - t
    t = time.time(); s_fast = fast_min_universe(C, p); t_fast = time.time() - t
    lam = lambda s: np.linalg.eigvalsh(C[np.ix_(s, s)])[-1]
    s_swap, v_swap = swap_search(C, s_fast, "min")
    rand = min(lam(rng.choice(n, p, replace=False).tolist()) for _ in range(20000))
    print("\nMIN first eigenvalue")
    print(f"  paper greedy      {lam(s_naive):.4f}   ({t_naive:.1f}s)")
    print(f"  fast greedy       {lam(s_fast):.4f}   ({t_fast:.2f}s)  same set: {set(s_fast)==set(s_naive)}")
    print(f"  + swap search     {v_swap:.4f}")
    print(f"  best of 20k random {rand:.4f}")
    print(f"  lower bound       {b['min_lower']:.4f}   -> gap <= {v_swap - b['min_lower']:.4f}")

    # --- MAX ----------------------------------------------------------------
    t = time.time(); m_fast = fast_max_universe(C, p, batch=10); t_fast = time.time() - t
    m_swap, mv_swap = swap_search(C, m_fast, "max")
    rand = max(lam(rng.choice(n, p, replace=False).tolist()) for _ in range(20000))
    print("\nMAX first eigenvalue")
    print(f"  fast backward     {lam(m_fast):.4f}   ({t_fast:.2f}s)")
    print(f"  + swap search     {mv_swap:.4f}")
    print(f"  best of 20k random {rand:.4f}")
    print(f"  upper bound       {b['max_upper']:.4f}   -> gap <= {b['max_upper'] - mv_swap:.4f}")

    # --- check fast backward vs paper backward on a size where naive is feasible
    n2 = 150
    C2 = C[:n2, :n2]
    t = time.time(); a = max_universe(C2, p); ta = time.time() - t
    t = time.time(); f = fast_max_universe(C2, p); tf = time.time() - t
    la = np.linalg.eigvalsh(C2[np.ix_(a, a)])[-1]
    lf = np.linalg.eigvalsh(C2[np.ix_(f, f)])[-1]
    print(f"\nBackward check on n={n2}: paper {la:.4f} ({ta:.1f}s) vs fast {lf:.4f} ({tf:.2f}s)")
