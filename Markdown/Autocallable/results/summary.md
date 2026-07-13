# Autocallable Proxy Experiment

Single-underlying autocallable note priced conditionally on not having
called before the current observation date.

Best method on this grid: `akima`.

| Method | Worst max relative error | Average p99 | Average MAE |
|---|---:|---:|---:|
| akima | 0.123% | 0.051% | 0.008583 |
| pchip | 0.135% | 0.050% | 0.008397 |

## Test Cases

| Case | Vol | Autocall barrier | Coupon barrier | Protection barrier | Coupon/obs |
|---|---:|---:|---:|---:|---:|
| base | 0.240 | 1.000 | 0.750 | 0.650 | 0.025 |
| high_vol | 0.340 | 1.000 | 0.750 | 0.600 | 0.025 |
| low_autocall | 0.240 | 0.920 | 0.750 | 0.650 | 0.018 |
| high_coupon | 0.280 | 1.000 | 0.750 | 0.650 | 0.040 |
| downside_heavy | 0.380 | 1.000 | 0.750 | 0.750 | 0.025 |

## Rate And Volatility Term Structures

Autocallables are observation-date products. Curve timing changes discount
factors, forward levels, and variances to each call/coupon date. In the
single-feature term-structure sensitivity study, front-loaded versus
back-loaded curves with the same total rate and total variance changed the
autocallable value by about 2.6%.

For a fixed deterministic curve, the one-dimensional spot proxy remains valid
at each observation index. If the proxy must generalize across curves, use
event-date summaries:

```text
discount factors to observation dates
forward distances to autocall/coupon/protection barriers
cumulative variance to observation dates
front/back volatility slope
```

Raw curve knots should be a fallback after these summaries fail residual tests.
