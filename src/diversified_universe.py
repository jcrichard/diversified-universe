"""
Greedy selection of the most / least diversified sub-universe
(Richard, "On the Selection of a Diversified Universe", Lyxor 2012).

Diversification is measured on the correlation (or covariance) matrix of the
selected assets. Two loss functions are available, matching the paper's code:

    "eig"  : first eigenvalue Lambda_0                         (method == 1)
    "flat" : mean((lambda_i - 1)^2), distance of the spectrum
             from the identity's flat spectrum                  (method == 2)

min_universe  -> forward selection  (start from the least correlated pair, add)
max_universe  -> backward elimination (start from the full universe, remove)

Both are greedy heuristics: they give no guarantee of reaching the global
optimum over all C(n, p) subsets. Use `brute_force` to check on small cases.
"""

from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dependence_matrix(returns, use="cor"):
    """Correlation ('cor') or covariance ('cov') matrix of a T x n returns array."""
    returns = np.asarray(returns, dtype=float)
    if use == "cor":
        return np.corrcoef(returns, rowvar=False)
    if use == "cov":
        return np.cov(returns, rowvar=False)
    raise ValueError("use must be 'cor' or 'cov'")


def loss(sub_matrix, criterion="eig"):
    """Diversification loss of a (sub)matrix. Lower = more diversified."""
    eigvals = np.linalg.eigvalsh(sub_matrix)  # ascending order
    if criterion == "eig":
        return eigvals[-1]
    if criterion == "flat":
        return np.mean((eigvals - 1.0) ** 2)
    raise ValueError("criterion must be 'eig' or 'flat'")


def explained_variance(sub_matrix):
    """pi_0: share of total variance explained by the first factor."""
    eigvals = np.linalg.eigvalsh(sub_matrix)
    return eigvals[-1] / eigvals.sum()


# ---------------------------------------------------------------------------
# Algorithm 1: minimum first eigenvalue universe (most diversified)
# ---------------------------------------------------------------------------
def min_universe(C, p, criterion="eig"):
    """
    Forward greedy selection.
      1. Start with the pair of assets with the lowest |correlation|.
      2. At each step add the asset that gives the smallest loss for the
         enlarged universe (by Cauchy interlacing, Lambda_0 can only increase
         when an asset is added, so this picks the smallest increase).
      3. Stop when the universe has p assets.

    Parameters
    ----------
    C : (n, n) correlation or covariance matrix of the full universe
    p : target number of assets (2 <= p <= n)

    Returns
    -------
    list of selected column indices, in order of inclusion
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    if not 2 <= p <= n:
        raise ValueError("p must satisfy 2 <= p <= n")

    # Step 1: least correlated pair (ignore the diagonal)
    A = np.abs(C).copy()
    np.fill_diagonal(A, np.inf)
    i, j = np.unravel_index(np.argmin(A), A.shape)
    selected = [int(i), int(j)]
    remaining = set(range(n)) - set(selected)

    # Step 2: greedy additions
    while len(selected) < p:
        best_asset, best_loss = None, np.inf
        for k in remaining:
            idx = selected + [k]
            l = loss(C[np.ix_(idx, idx)], criterion)
            if l < best_loss:
                best_asset, best_loss = k, l
        selected.append(best_asset)
        remaining.remove(best_asset)

    return selected


# ---------------------------------------------------------------------------
# Algorithm 2: maximum first eigenvalue universe (least diversified)
# ---------------------------------------------------------------------------
def max_universe(C, p, criterion="eig"):
    """
    Backward greedy elimination.
      1. Start with the full universe of n assets.
      2. At each step remove the asset whose deletion keeps the loss as high
         as possible (i.e. minimizes Lambda_0^n - Lambda_0^(n-1)).
      3. Stop when p assets remain.

    Returns
    -------
    list of the p remaining column indices (sorted)
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    if not 1 <= p <= n:
        raise ValueError("p must satisfy 1 <= p <= n")

    selected = list(range(n))
    while len(selected) > p:
        best_drop, best_loss = None, -np.inf
        for k in selected:
            idx = [a for a in selected if a != k]
            l = loss(C[np.ix_(idx, idx)], criterion)
            if l > best_loss:
                best_drop, best_loss = k, l
        selected.remove(best_drop)

    return sorted(selected)


# ---------------------------------------------------------------------------
# Optimal size p* (Section 2.1): minimise pi_0 on a test set
# ---------------------------------------------------------------------------
def optimal_universe(C_train, C_test, p_range=None, criterion="eig"):
    """
    For each p, build the min universe on the training matrix, then measure
    pi_0 of that universe on the test matrix. Return the p minimising it.

    Returns
    -------
    (best_p, best_assets, results) where results maps p -> (assets, pi0_train, pi0_test)
    """
    n = np.asarray(C_train).shape[0]
    p_range = p_range or range(2, n + 1)
    results = {}
    for p in p_range:
        idx = min_universe(C_train, p, criterion)
        sub = np.ix_(idx, idx)
        results[p] = (idx, explained_variance(C_train[sub]),
                      explained_variance(C_test[sub]))
    best_p = min(results, key=lambda q: results[q][2])
    return best_p, results[best_p][0], results


# ---------------------------------------------------------------------------
# Exhaustive search (for validating the greedy algorithms on small n)
# ---------------------------------------------------------------------------
def brute_force(C, p, criterion="eig"):
    """Exact min and max loss over all C(n, p) subsets. Small n only."""
    n = C.shape[0]
    best_min, best_max = (np.inf, None), (-np.inf, None)
    for idx in combinations(range(n), p):
        l = loss(C[np.ix_(idx, idx)], criterion)
        if l < best_min[0]:
            best_min = (l, list(idx))
        if l > best_max[0]:
            best_max = (l, list(idx))
    return best_min, best_max


# ---------------------------------------------------------------------------
# Demo on synthetic multi-asset data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 4 correlated blocks (e.g. equities, bonds, commodities, FX) + noise
    T, block_sizes = 1000, [8, 5, 6, 5]
    factors = rng.standard_normal((T, len(block_sizes)))
    loadings = [0.9, 0.6, 0.4, 0.3]
    cols = []
    for b, (size, beta) in enumerate(zip(block_sizes, loadings)):
        for _ in range(size):
            cols.append(beta * factors[:, b] + rng.standard_normal(T))
    R = np.column_stack(cols)
    n = R.shape[1]

    train, test = R[: T // 2], R[T // 2:]
    C_train, C_test = dependence_matrix(train), dependence_matrix(test)

    p = 5
    idx_min = min_universe(C_train, p)
    idx_max = max_universe(C_train, p)
    lam = lambda idx: loss(C_train[np.ix_(idx, idx)])

    print(f"n = {n} assets, p = {p}")
    print(f"Min universe {sorted(idx_min)}  Lambda_0 = {lam(idx_min):.3f}")
    print(f"Max universe {idx_max}  Lambda_0 = {lam(idx_max):.3f}")

    (bf_min, bf_min_idx), (bf_max, bf_max_idx) = brute_force(C_train, p)
    print(f"Exhaustive min Lambda_0 = {bf_min:.3f} {bf_min_idx}")
    print(f"Exhaustive max Lambda_0 = {bf_max:.3f} {bf_max_idx}")

    best_p, best_idx, res = optimal_universe(C_train, C_test)
    print(f"\nOptimal size p* = {best_p}, pi_0 (test) = {res[best_p][2]:.1%}"
          f"  (1/p* = {1 / best_p:.1%})")
