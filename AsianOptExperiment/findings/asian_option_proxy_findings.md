# Asian Option Proxy Findings

This note summarizes what we learned from the arithmetic Asian option proxy
experiments. The benchmark product is a monthly observed arithmetic Asian call
under GBM with 12 fixings over 1 year.

There is no arithmetic Asian closed form in the experiment. Monte Carlo is used
for both training labels and validation benchmarks, with stronger variance
reduction and more paths for the benchmark.

## Main Conclusion

The strongest finding is:

> For GBM Asian options, transform the state before fitting. The raw 2D state
> `(spot, running sum)` is the right Markov state, but not the best regression
> coordinate.

For a fixed observation date, the future Asian payoff can be rewritten with an
adjusted strike:

```text
K_adj = (N K - running_sum_before - spot_today) / future_fixings
```

Under GBM scaling, the continuation value is effectively:

```text
spot_today * f(K_adj / spot_today)
```

So the best proxy is almost one-dimensional after the adjusted-moneyness
transformation. This was much better than fitting a generic 2D surface in
`(spot, running average)`.

## Best Current Method

The current default Asian method is:

```text
product:                  monthly arithmetic Asian call
fixings:                  12 observations over 1 year
state:                    spot today, running sum before today
state transform:           adjusted d1-like moneyness
training labels:           shifted antithetic MC
training scenarios/fit:    about 10,000,000
benchmark labels:          500,000 paths per validation state
benchmark variance cut:    geometric Asian control variate
target 1:                  log(value / spot)
target 2:                  log(time value / spot)
default fitter:            Chebyshev degree 19 for both targets
ITM switch:                use linear baseline + time value once
                           linear baseline > 5% of spot
diagnostic metric:         abs(proxy - benchmark) / max(benchmark, 0.01)
```

The benchmark estimator is:

```text
shifted antithetic MC
+ likelihood-ratio correction
+ exact discrete geometric Asian control variate
```

## Monthly 12-Fixing Results

The best method was the adjusted-moneyness hybrid.

| Day Index | Remaining Fixings | Max % Error | P99 % Error | P95 % Error | MAE | Max Abs Error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11 | 0.523% | 0.485% | 0.337% | 0.008815 | 0.067432 |
| 3 | 8 | 2.454% | 2.341% | 2.048% | 0.008520 | 0.084497 |
| 6 | 5 | 2.129% | 1.932% | 1.532% | 0.006248 | 0.070201 |
| 9 | 2 | 1.090% | 0.946% | 0.744% | 0.001406 | 0.035703 |
| 11 | 0 | 0.000% | 0.000% | 0.000% | 0.000000 | 0.000000 |

Aggregate result:

```text
worst max error:  2.454%
average p99:      1.141%
average MAE:      0.004998
```

## Methods Tested

| Method | Worst Max % Error | Avg P99 % Error | Avg MAE |
|---|---:|---:|---:|
| Adjusted-moneyness hybrid | 2.454% | 1.141% | 0.004998 |
| Forward 2D hybrid, tensor | 119.899% | 26.614% | 0.048697 |
| Forward 2D log, Halton | 305.232% | 102.239% | 2.408076 |
| Forward 2D log, tensor | 344.357% | 75.119% | 1.836864 |
| Forward 2D hybrid, Halton | 364.074% | 79.629% | 0.125482 |
| Raw 2D log Chebyshev | 378.099% | 71.251% | 0.960724 |

## What We Learned

### 1. A generic 2D fit is fragile

The raw 2D log-Chebyshev fit can work near inception, but it fails badly later.
The payoff boundary is diagonal in `(spot, running average)`, and the ITM region
becomes almost exactly linear. A smooth global 2D polynomial spends too much
capacity fighting this geometry.

### 2. Better 2D state sampling was not enough

We tried boundary-enriched grids and Halton low-discrepancy sampling in
forward-average coordinates. These sampled more OTM/ITM and boundary states, but
they still did not beat the structural adjusted-moneyness reduction.

This does not mean low-discrepancy state sampling is useless. It means that for
this GBM Asian setup, the coordinate transform matters more than state sampling
density.

### 3. The known linear wing should be used directly

When `K_adj <= 0`, the call payoff is always positive and the value is linear in
the expected future average. At terminal date, the payoff is exact. The default
proxy handles both regions analytically instead of asking the regression to
learn them.

### 4. The remaining error is near the exercise boundary

The worst errors are small boundary ripples where values are low but not
negligible. With the current benchmark, the monthly 12-fixing test stays inside
the 3-5% target range.

## Default Entry Point

The standalone parent-level script is:

```powershell
python AsianMain.py
```

It runs only the current default method and writes:

```text
AsianOptExperiment/default_run/asian_default_results.csv
```

The broader exploratory script and generated comparison artifacts remain under:

```text
AsianOptExperiment/
```
