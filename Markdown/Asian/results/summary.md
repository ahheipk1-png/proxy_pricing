# Asian Option Proxy Experiment

Monthly arithmetic Asian call under GBM with 12 fixings over 1 year.
State is `(spot today, running sum before today)`.

No arithmetic Asian closed form is used. Each fitted daily proxy uses about 10,000,000 shifted-antithetic MC training scenarios spread over
that fit's training grid. Benchmarks use 500,000 shifted-antithetic
MC scenarios per validation state, with a discrete geometric Asian control variate.

Relative errors below use a denominator floor of `0.01`, so near-zero prices do not dominate
the percentages purely through division by a tiny value.

## Methods Compared

| Method | Description | Worst Max % Error | Avg P99 % Error | Avg P95 % Error | Avg MAE | Max Abs Error |
|---|---|---:|---:|---:|---:|---:|
| Adjusted-moneyness hybrid | 1D adjusted-moneyness hybrid: log value degree 19, log time-value degree 19 | 2.454% | 1.141% | 0.932% | 0.004998 | 0.084497 |
| Forward 2D hybrid, tensor | Forward-coordinate 2D hybrid degrees 9/9, boundary-enriched tensor grid | 119.899% | 26.614% | 21.813% | 0.048697 | 4.988992 |
| Forward 2D log, Halton | Forward-coordinate 2D log Chebyshev degree 9, Halton global/boundary state sampling | 305.232% | 102.239% | 58.103% | 2.408076 | 72.288904 |
| Forward 2D log, tensor | Forward-coordinate 2D log Chebyshev degree 9, boundary-enriched tensor grid | 344.357% | 75.119% | 58.488% | 1.836864 | 38.944097 |
| Forward 2D hybrid, Halton | Forward-coordinate 2D hybrid degrees 9/9, Halton global/boundary state sampling | 364.074% | 79.629% | 26.328% | 0.125482 | 12.333737 |
| Naive 2D extension | 2D log Chebyshev degree 7 | 378.099% | 71.251% | 47.906% | 0.960724 | 25.396190 |

## Best Overall Method By Day: Adjusted-moneyness hybrid

| Day Index | Remaining Fixings | Max % Error | P99 % Error | P95 % Error | MAE | Max Abs Error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11 | 0.523% | 0.485% | 0.337% | 0.008815 | 0.067432 |
| 3 | 8 | 2.454% | 2.341% | 2.048% | 0.008520 | 0.084497 |
| 6 | 5 | 2.129% | 1.932% | 1.532% | 0.006248 | 0.070201 |
| 9 | 2 | 1.090% | 0.946% | 0.744% | 0.001406 | 0.035703 |
| 11 | 0 | 0.000% | 0.000% | 0.000% | 0.000000 | 0.000000 |

## What Changed Versus The Naive 2D Proxy

- The raw 2D log-Chebyshev extension works at inception but fails later because the payoff
  boundary is diagonal in `(spot, running average)` and the ITM wing becomes almost exactly
  linear.
- The forward-coordinate methods sample and fit in expected-average moneyness plus
  spot/history ratio, which directly tests enriched OTM/ITM state sampling.
- The Halton variants use low-discrepancy state sampling in those same forward coordinates,
  with half the points in a global box and half near the payoff boundary.
- For fixed day under GBM, the Asian continuation payoff can be rewritten with an adjusted
  strike: `K_adj = (N K - running_sum_before - spot) / future_fixings`.
- GBM scaling implies the continuation value can be represented by
  `spot * f(K_adj / spot)` for each remaining tenor, so the best proxy is fitted in
  adjusted moneyness rather than raw state coordinates.
- The hybrid uses a log-price fit near/OTM and switches to `linear baseline + log time value`
  once the linear baseline exceeds 5.0% of spot.
- The exact terminal payoff and the `K_adj <= 0` linear region are handled analytically.

## Files

- Method summary CSV: `asian_proxy_method_results.csv`
- Validation detail CSV: `asian_proxy_validation_details.csv`
- Hybrid plots: `plots/asian_day_*.png`

## Short Conclusion

The best overall method in this run is `adjusted_moneyness_hybrid`, with tested max error
2.454% and average p99 error
1.141%.
