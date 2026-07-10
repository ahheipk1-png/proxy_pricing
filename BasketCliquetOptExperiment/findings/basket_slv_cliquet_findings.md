# Generalized Three-Underlying SLV Basket Cliquet Findings

## Contract variants

All variants use three SLV underlyings, 12 monthly resets, notional 100,
local floor/cap `-2% / 4%`, global floor/cap `0% / 20%`, and sum aggregation.

The generalized monthly coupon cases are:

| Variant | Coupon rule |
|---|---|
| `basket_return` | Clip the equal-weight average return |
| `weighted_average` | Clip a non-equal weighted average of individual returns |
| `basket_ratio` | Clip the non-equal weighted basket-ratio return |
| `average_clipped` | Average the three individually clipped returns |
| `second_worst` | Clip the second-worst underlying return |
| `worst_of` | Clip the worst-performing underlying return |
| `best_of` | Clip the best-performing underlying return |
| `spread_bonus` | Weighted-average coupon minus spread coupon plus bonus trigger |

## Model and state

Each asset has its own stochastic variance and bounded local-volatility
leverage. Market Brownian drivers are correlated, while each asset has a
negative spot/variance shock correlation.

At reset dates the Markov state is seven-dimensional:

```text
(accrued return, S1, S2, S3, v1, v2, v3)
```

The proxy features include lower and upper global-bound cushions, coupon
moments, floor/cap masses, raw log spots and log variances, weighted/min/max
and dispersion summaries, and PCA coordinates of log spots and log variances.

## Training and benchmark

The latest run uses grouped labels:

- 1,009 low-discrepancy market states;
- 17 accrued-return layers per market state;
- 32,768 Sobol/LR paths per market state after component rounding;
- 31 independent validation states;
- 524,288 Sobol/LR benchmark paths per validation state;
- two full-truncation Euler steps per monthly coupon period;
- antithetic Sobol points;
- an 8-component likelihood-ratio mixture over common market and dispersion
  directions.

The grouped-label trick matters. For a fixed market state, the simulated future
coupon sums can be reused for many initial accrued returns:

```text
V(a, M) = discount * E[N * clip(a + future_coupon_sum, floor, cap) | M].
```

So one path simulation for market state `M` prices the whole accrued-return
strip. This reduced training noise without multiplying runtime by the number of
accrued layers.

## Methods tried

The experiment compared:

- local quadratic regression on summary features;
- local linear and quadratic regression on the full state;
- sparse anisotropic Chebyshev regression;
- an adaptive local/sparse blend;
- moment-normal anchored sparse Chebyshev residuals;
- accrued-return PCHIP plus nearest-neighbor market-state interpolation;
- payoff-aware PCA and spread features.

The anchored residual and PCHIP/kNN ideas were useful diagnostics but not the
best production candidates in this run. The moment-normal anchor was too
Gaussian for bounded order-statistic tails, and PCHIP/kNN overpredicted some
near-floor tail states.

## Latest generalized results

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

## Conclusion

The grouped-label/PCA methodology is a real improvement for basket-like coupons:
`basket_return` is within the 5-8% target and `average_clipped` is just inside
that range. It is not yet a generic 5-8% solution for all generalized basket
cliquets.

The hard cases are the order-statistic coupons, especially `best_of` and
`worst_of`. Their near-floor values are rare events in a bounded sum of clipped
maximum or minimum returns. Small absolute misses create large percentages, but
some misses are still economically meaningful enough that the method should not
be called solved.

The next credible path is adaptive state enrichment targeted at failed
order-statistic neighborhoods, or a larger path-level model that learns the
bounded order-statistic distribution more directly. A single fixed sparse
polynomial/spline method is not enough for every generalized basket cliquet
style tested here.
