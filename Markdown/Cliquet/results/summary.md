# Cliquet Option Proxy Experiment

Monthly cliquet under GBM with 12 local return observations over 1 year.
Payoff is notional times the globally floored/capped sum of locally floored/capped returns.

Local floor/cap: -2.0% / 4.0%.
Global floor/cap: 0.0% / 20.0%.

Each fitted proxy uses about 10,000,000 MC training scenarios spread over
its training state grid. Benchmarks use 500,000 antithetic MC
scenarios per validation state with an exact clipped-return-sum control variate.

## Methods Compared

| Method | Description | Worst Max % Error | Avg P99 % Error | Avg P95 % Error | Avg MAE | Max Abs Error |
|---|---|---:|---:|---:|---:|---:|
| `logit_z_boundary_d19` | Bounded logit Chebyshev degree 19 on expected-total z feature, boundary grid | 3.593% | 0.928% | 0.500% | 0.003194 | 0.029218 |
| `logit_accrued_boundary_d9` | Bounded logit Chebyshev degree 9 on raw accrued return, boundary grid | 4.922% | 1.012% | 0.779% | 0.011240 | 0.202753 |
| `logit_z_boundary_d11` | Bounded logit Chebyshev degree 11 on expected-total z feature, boundary grid | 5.775% | 1.336% | 0.894% | 0.006533 | 0.096313 |
| `logit_z_halton_d11` | Bounded logit Chebyshev degree 11 on expected-total z feature, Halton/boundary grid | 8.660% | 1.749% | 1.083% | 0.005640 | 0.076054 |
| `logit_z_uniform_d9` | Bounded logit Chebyshev degree 9 on expected-total z feature, uniform grid | 28.204% | 4.739% | 1.649% | 0.014677 | 0.230737 |
| `logit_cushion_2d_boundary_d5` | Bounded logit 2D Chebyshev degree 5 on floor/cap cushion features, boundary grid | 28.735% | 4.721% | 3.438% | 0.019067 | 0.317438 |
| `direct_accrued_uniform_d7` | Direct value Chebyshev degree 7 on raw accrued return, uniform grid | 87.233% | 13.953% | 5.434% | 0.004825 | 0.048928 |

## Best Overall Method By Day: `logit_z_boundary_d19`

| Day Index | Remaining Periods | Max % Error | P99 % Error | P95 % Error | MAE | Max Abs Error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 0.033% | 0.033% | 0.033% | 0.002569 | 0.002569 |
| 3 | 9 | 0.500% | 0.451% | 0.206% | 0.003297 | 0.015249 |
| 6 | 6 | 1.612% | 0.991% | 0.566% | 0.003318 | 0.015805 |
| 9 | 3 | 3.593% | 3.167% | 1.694% | 0.006787 | 0.029218 |
| 12 | 0 | 0.000% | 0.000% | 0.000% | 0.000000 | 0.000000 |

## Main Takeaways

- The natural Markov state is the accrued clipped return. Spot is not needed at reset dates
  because future GBM returns are scale-invariant.
- Raw value regression can work, but bounded-logit targets are more stable because the
  cliquet payoff has known global floor/cap bounds.
- The best feature in this run is the expected-total z-score: accrued return plus expected
  future clipped return, normalized by future clipped-return volatility.
- Exact tails are used when the remaining local floors/caps imply the global floor or cap
  is already locked in.

## Files

- Method summary CSV: `cliquet_proxy_method_results.csv`
- Validation detail CSV: `cliquet_proxy_validation_details.csv`
- Best-method plots: `plots/cliquet_day_*.png`

## Short Conclusion

The best overall method is `logit_z_boundary_d19`, with tested max error 3.593% and average p99 error 0.928%.
