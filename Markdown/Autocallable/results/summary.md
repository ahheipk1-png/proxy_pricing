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
