# Selecting a diversified universe

Code and experiments around J.-C. Richard, *On the Selection of a Diversified Universe* (Lyxor, 2012):
reimplementation of the paper's algorithms, faster versions, alternative selection criteria,
and a Monte Carlo comparison. All methods are pure in/out selection: no portfolio weights.

## Methods

| Method | Objective | Algorithm |
|---|---|---|
| Paper (published) | min first eigenvalue Λ₀ | Forward greedy from the least correlated pair |
| Paper + swap | min Λ₀ | Paper greedy + 1-in/1-out swap search |
| Paper, max universe | max Λ₀ | Backward elimination from the full universe |
| Max diversity \|ρ\| | min Σ \|ρᵢⱼ\| over selected pairs | Greedy + swap + random restarts; exact integer program for small n |
| Max diversity ρ | min Σ ρᵢⱼ (signed) | Same |
| Log-det | max log det of the selected correlation matrix | Pivoted Cholesky greedy + swap + restarts |

## Files

- `src/diversified_universe.py`: paper's min and max algorithms, direct translation of the GAUSS code
- `src/diversified_universe_fast.py`: scalable versions (secular-equation forward step, Rayleigh-bound backward step), swap search, optimality bounds
- `src/binary_selection.py`: max diversity and log-det optimizers, exact MILP check, `describe()` metrics
- `src/montecarlo_500.py`: Monte Carlo experiment
- `results/montecarlo_500_results.csv`: raw results (100 simulations × 3 values of p × 6 methods)
- `results/summary_tables.md`: averaged results

## Usage

```python
import numpy as np
from src.binary_selection import diversity_optimize, logdet_optimize, describe
from src.diversified_universe_fast import fast_min_universe, swap_search

C = np.corrcoef(returns, rowvar=False)          # returns: T x n array
S_div, _ = diversity_optimize(C, p=20, kind="abs")
S_ld, _  = logdet_optimize(C, p=20)
S_pap    = fast_min_universe(C, p=20)
describe(C, S_div)                               # Λ₀, π₀, mean ρ, mean |ρ|, effective factors, log-det
```

Run the experiment: `python src/montecarlo_500.py 100` (about 4 minutes on one core).
Requirements: `numpy`, `scipy`, `pandas`.

## Monte Carlo experiment

100 random "true" 500 × 500 correlation matrices (2–30 signed factors, 3–25 sectors, a market factor
from absent to dominant, up to 30% of assets flipped to create negative correlations).
For each, 750 days of returns are simulated; methods select on the **sample** matrix and are scored
on the **true** matrix.

True Λ₀ (lower is better) and share of simulations where each method was best:

| Method | p = 10 | p = 20 | p = 50 | Wins (10 / 20 / 50) |
|---|---|---|---|---|
| Paper (published) | 1.336 | 1.782 | 3.278 | 10% / 17% / 21% |
| Paper + swap | 1.314 | 1.757 | **3.236** | 15% / 29% / **54%** |
| Max diversity \|ρ\| | **1.293** | **1.750** | 3.387 | 38% / **40%** / 25% |
| Log-det | 1.297 | 1.818 | 4.132 | **42%** / 15% / 0% |
| Max diversity ρ (signed) | 3.760 | 6.057 | 12.031 | 0% / 0% / 0% |
| Random | 2.861 | 4.926 | 10.668 | – |

### Findings

- Every optimized method is far better than random; differences among the good methods are a few percent of Λ₀.
- Max diversity \|ρ\| beats the published paper method on the paper's own criterion for p = 10 and 20 (in 73% and 65% of simulations, significant), and is the fastest method.
- For p = 50, the paper's approach wins on Λ₀, especially with swap search.
- Max diversity \|ρ\| and log-det are best on whole-spectrum measures (mean \|ρ\|, effective number of factors, log-det) at every p.
- Signed max diversity is a trap: it picks strongly negatively correlated pairs, creating one large long/short factor, worse than random on Λ₀.
- All methods look better in-sample than on the true matrix (optimism of 0.06–0.13 in Λ₀ at p = 20); paper + swap is the most optimistic.

Caveat: conclusions depend on the simulated correlation structures; they should be confirmed on real data.
# diversified-universe
# diversified-universe
# diversified-universe
