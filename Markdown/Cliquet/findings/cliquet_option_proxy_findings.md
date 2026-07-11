# Cliquet Option Proxy Findings

This note summarizes the monthly cliquet proxy experiments. The benchmark product
is a 1-year monthly cliquet under GBM with 12 locally capped/floored returns.

The payoff is:

```text
100 * clip(sum_i clip(R_i, local_floor, local_cap), global_floor, global_cap)
```

with:

```text
local floor / cap:  -2% / 4%
global floor / cap:  0% / 20%
```

## Main Conclusion

The strongest finding is:

> At reset dates, the cliquet proxy is an accumulator problem, not a spot problem.

Future GBM returns are scale-invariant, so current spot is not needed once the
state is measured immediately after an observation/reset. The relevant state is
the accrued sum of clipped local returns.

The best feature was not raw accrued return by itself, but a normalized
expected-total feature:

```text
z = (accrued_return + E[future clipped return sum] - global_midpoint)
    / stdev(future clipped return sum)
```

This puts different reset dates on comparable scales and lines up the regression
with the global floor/cap transition region.

## Best Current Method

The current default cliquet method is:

```text
product:                  monthly cliquet
periods:                  12 observations over 1 year
state:                    accrued clipped return
state feature:             expected-total z-score
training grid:             boundary-enriched accrued-return grid
training labels:           antithetic GBM Monte Carlo
training scenarios/fit:    about 10,000,000
benchmark labels:          500,000 paths per validation state
benchmark variance cut:    exact clipped-return-sum control variate
target:                   bounded logit of normalized value
default fitter:            Chebyshev degree 19
diagnostic metric:         abs(proxy - benchmark) / max(benchmark, 0.01)
```

The known exact regions are used directly. If the remaining local floors/caps
already force the global floor or global cap, the proxy returns the exact
discounted bound instead of asking the regression to learn the flat tail.

## Benchmark

There is no closed form for the globally clipped cliquet payoff in this
experiment. The benchmark is:

```text
antithetic GBM Monte Carlo
+ 500,000 paths per validation state
+ exact control variate for the future clipped-return sum
```

The control variate uses closed-form first and second moments of one clipped
lognormal return:

```text
clip(exp(X) - 1, local_floor, local_cap)
```

where `X` is normally distributed under GBM.

## Results

| Day Index | Remaining Periods | Max % Error | P99 % Error | P95 % Error | MAE | Max Abs Error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 0.033% | 0.033% | 0.033% | 0.002569 | 0.002569 |
| 3 | 9 | 0.500% | 0.451% | 0.206% | 0.003297 | 0.015249 |
| 6 | 6 | 1.612% | 0.991% | 0.566% | 0.003318 | 0.015805 |
| 9 | 3 | 3.593% | 3.167% | 1.694% | 0.006787 | 0.029218 |
| 12 | 0 | 0.000% | 0.000% | 0.000% | 0.000000 | 0.000000 |

Aggregate result:

```text
worst max error:  3.593%
average p99:      0.928%
average MAE:      0.003194
```

## Methods Tested

| Method | Worst Max % Error | Avg P99 % Error | Avg MAE |
|---|---:|---:|---:|
| bounded logit, expected-total z, boundary grid, degree 19 | 3.593% | 0.928% | 0.003194 |
| bounded logit, accrued return, boundary grid, degree 9 | 4.922% | 1.012% | 0.011240 |
| bounded logit, expected-total z, boundary grid, degree 11 | 5.775% | 1.336% | 0.006533 |
| bounded logit, expected-total z, Halton/boundary grid, degree 11 | 8.660% | 1.749% | 0.005640 |
| bounded logit, expected-total z, uniform grid, degree 9 | 28.204% | 4.739% | 0.014677 |
| bounded logit, 2D floor/cap cushion features, boundary grid, degree 5 | 28.735% | 4.721% | 0.019067 |
| direct value, accrued return, uniform grid, degree 7 | 87.233% | 13.953% | 0.004825 |

## What We Learned

### 1. Bounded targets matter

The cliquet has hard global floor/cap bounds. Fitting raw price directly can
produce small average dollar error but poor max percentage error near the payoff
transition. A bounded logit target respects the known range.

### 2. Boundary-enriched state sampling matters

Uniform accumulator grids miss too much of the floor/cap transition geometry.
The boundary-enriched grid deliberately samples around the estimated global
floor and global cap transition centers.

### 3. Extra 2D features did not help here

The 2D floor/cap cushion feature set is conceptually reasonable, but for this
product it did worse than the single expected-total z feature. The state is
effectively one-dimensional at reset dates.

### 4. Low-discrepancy state sampling was not enough by itself

The Halton/boundary state grid was competitive but did not beat the deterministic
boundary-enriched grid. For this product, putting points near the right transition
regions mattered more than using a low-discrepancy sequence.

## Default Entry Point

The standalone parent-level script is:

```powershell
python CliquetMain.py
```

It runs only the current default method and writes:

```text
CliquetOptExperiment/default_run/cliquet_default_results.csv
```

The exploratory script and generated comparison artifacts remain under:

```text
CliquetOptExperiment/
```
