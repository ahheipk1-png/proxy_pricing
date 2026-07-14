# Single Generic Exotic Proxy Study

This study implements the requested data-driven product pipeline:

```text
Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff
```

Existing product-specific main scripts were not replaced. This is a new
generic proxy layer for exotic payoff configurations.

## Setup

- configurations priced: `106`
- train states per configuration: `241` common spot-scale states
- validation states per configuration: `101` shifted spot-scale states
- train paths per state label: `65,536` low-discrepancy antithetic paths
- benchmark paths per validation state: `65,536` independent paths
- proxy candidates: direct/log linear interpolation and direct/log PCHIP interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `292.7`

## Accuracy Summary

- PASS: `106`
- WATCH: `0`
- REVIEW: `0`

| Family | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Barrier | 24 | 4.408% | 1.067% | 3.785% | 0.002561 | 24 | 0 | 0 |
| Binary | 15 | 9.410% | 1.249% | 3.228% | 0.008977 | 15 | 0 | 0 |
| Himalayan | 6 | 0.059% | 0.028% | 0.033% | 0.000036 | 6 | 0 | 0 |
| Lookback | 16 | 7.382% | 0.037% | 0.134% | 0.000032 | 16 | 0 | 0 |
| Rainbow | 27 | 0.027% | 0.016% | 0.021% | 0.000014 | 27 | 0 | 0 |
| YieldSeeker | 18 | 0.006% | 0.003% | 0.006% | 0.000550 | 18 | 0 | 0 |

## Worst Cases

| Case | Family | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| binary_1_arithmetic_average_double_digital_weighted_basket_1 | Binary | log_linear | 9.410% | 2.421% | 0.013660 | PASS |
| binary_1_arithmetic_average_range_digital_weighted_basket_1 | Binary | log_linear | 9.410% | 2.421% | 0.013660 | PASS |
| lookback_1_minimum_fixed_strike | Lookback | linear | 7.382% | 0.458% | 0.000377 | PASS |
| barrier_1_lower_knock_in_discrete_cash_weighted_basket_1 | Barrier | log_pchip | 4.408% | 3.810% | 0.006075 | PASS |
| barrier_1_upper_knock_in_continuous_cash_weighted_basket_1 | Barrier | log_linear | 4.076% | 2.633% | 0.007733 | PASS |
| barrier_1_lower_knock_in_continuous_cash_weighted_basket_1 | Barrier | pchip | 4.028% | 3.860% | 0.006362 | PASS |
| binary_1_arithmetic_average_asset_or_nothing_weighted_basket_1 | Binary | log_linear | 4.004% | 3.193% | 0.007966 | PASS |
| binary_1_arithmetic_average_cash_or_nothing_weighted_basket_1 | Binary | log_linear | 4.004% | 3.309% | 0.007927 | PASS |
| barrier_1_upper_knock_in_discrete_cash_weighted_basket_1 | Barrier | log_linear | 3.896% | 3.644% | 0.005310 | PASS |
| barrier_1_lower_knock_in_continuous_call_weighted_basket_1 | Barrier | linear | 2.648% | 1.265% | 0.000162 | PASS |
| binary_1_spot_cash_or_nothing_weighted_basket_1 | Binary | linear | 1.808% | 1.072% | 0.008762 | PASS |
| binary_1_spot_asset_or_nothing_weighted_basket_1 | Binary | linear | 1.706% | 1.005% | 0.008729 | PASS |
| barrier_1_upper_knock_in_discrete_put_weighted_basket_1 | Barrier | log_pchip | 1.645% | 1.199% | 0.000080 | PASS |
| binary_1_tail_average_cash_or_nothing_weighted_basket_1 | Binary | linear | 1.584% | 1.426% | 0.009260 | PASS |
| barrier_1_upper_knock_in_continuous_put_weighted_basket_1 | Barrier | linear | 1.550% | 1.075% | 0.000115 | PASS |

## Product Building Blocks Covered

- observations: spot, arithmetic average, tail average, lookback
- performance: fixed notional, fixed unit, relative
- ranking: identity, weighted basket, order statistic
- transformation: floor, cap, combined clamp
- aggregation: average, sum, compounded
- payoff families: rainbow, Himalayan, yield seeker, lookback, barrier, binary

## Files

- case CSV: `C:\codex_proj\proxy_pricing\SingleExoticOptExperiment\results\single_generic_exotic_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\SingleExoticOptExperiment\results\single_generic_exotic_proxy_details.csv`
- plot: `C:\codex_proj\proxy_pricing\SingleExoticOptExperiment\results\single_generic_exotic_proxy_accuracy.png`
