# Generalized three-underlying SLV basket cliquet experiment

The payoff cases are based on the generalized multi-asset cliquet note:
weighted-average, basket-ratio, order-statistic, and spread/bonus local coupons.
All cases use sum aggregation and the same global floor/cap payoff.

Training uses 1,009 low-discrepancy market states, 17 accrued-return layers per
market state, and grouped Sobol/LR labels so one simulated future-coupon
distribution prices all accrued layers for that market state.

The SLV path sampler uses antithetic Sobol points and an 8-component
likelihood-ratio mixture over common market and dispersion directions. The proxy
feature set includes lower/upper payoff cushions and PCA coordinates of log
spots and log variances.

Conclusion for fitted-only proxies: this is a useful improvement for
basket-like coupons, but not yet a universal 5-8% method for all generalized
basket cliquets. Order-statistic coupons remain the hard cases unless the
cached Sobol/LR safety proxy is used.

| Variant | Best method | Worst max relative error | Average p99 | Average MAE |
|---|---|---:|---:|---:|
| `basket_return` | `adaptive_blend` | 6.448% | 3.060% | 0.021583 |
| `weighted_average` | `adaptive_blend` | 11.280% | 4.946% | 0.023668 |
| `basket_ratio` | `adaptive_blend` | 19.942% | 7.946% | 0.024628 |
| `average_clipped` | `adaptive_blend` | 7.967% | 4.838% | 0.039313 |
| `second_worst` | `local_summary_quadratic` | 11.326% | 7.078% | 0.031038 |
| `worst_of` | `local_summary_quadratic` | 29.002% | 13.103% | 0.015456 |
| `best_of` | `local_summary_quadratic` | 66.880% | 24.321% | 0.043633 |
| `spread_bonus` | `local_summary_quadratic` | 15.911% | 12.466% | 0.025340 |

## Cached Sobol/LR safety proxy

The source now includes `sobol_mc_proxy`, a slower cached safety layer for hard
generalized basket cliquets. A no-file validation check on 31 states at reset
months 3, 6, and 9 compared 65,536-path safety prices with 262,144-path
benchmarks:

| Variant | Worst max relative error |
|---|---:|
| `basket_ratio` | 3.0% |
| `second_worst` | 4.4% |
| `worst_of` | 11.2% |
| `best_of` | 2.9% |
| `spread_bonus` | 11.0% |

This clears the 12% target on the tested hard cases, at the cost of being slower
than a pure fitted regression proxy.
