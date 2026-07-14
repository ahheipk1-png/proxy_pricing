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
- benchmark paths per validation state: `65,536` independent low-discrepancy antithetic paths
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `260.2`

## Accuracy Summary

- PASS: `70`
- WATCH: `20`
- REVIEW: `16`

| Family | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Barrier | 24 | 22.066% | 6.140% | 16.982% | 0.030574 | 13 | 3 | 8 |
| Binary | 15 | 22.094% | 9.588% | 16.029% | 0.090554 | 2 | 7 | 6 |
| Himalayan | 6 | 4.853% | 3.941% | 4.852% | 0.009537 | 6 | 0 | 0 |
| Lookback | 16 | 8.877% | 2.388% | 8.856% | 0.005926 | 13 | 1 | 2 |
| Rainbow | 27 | 6.230% | 4.589% | 6.126% | 0.008507 | 18 | 9 | 0 |
| YieldSeeker | 18 | 0.043% | 0.023% | 0.034% | 0.005175 | 18 | 0 | 0 |

## Worst Cases

| Case | Family | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| binary_1_arithmetic_average_asset_or_nothing_weighted_basket_1 | Binary | log_linear | 22.094% | 16.353% | 0.041982 | REVIEW |
| barrier_1_upper_knock_in_continuous_cash_weighted_basket_1 | Barrier | pchip | 22.066% | 20.107% | 0.045082 | REVIEW |
| binary_1_arithmetic_average_double_digital_weighted_basket_1 | Binary | log_linear | 21.247% | 15.362% | 0.070743 | REVIEW |
| binary_1_arithmetic_average_range_digital_weighted_basket_1 | Binary | log_linear | 21.247% | 15.362% | 0.070743 | REVIEW |
| binary_1_arithmetic_average_cash_or_nothing_weighted_basket_1 | Binary | log_linear | 21.034% | 15.890% | 0.041415 | REVIEW |
| barrier_1_lower_knock_in_continuous_cash_weighted_basket_1 | Barrier | nearest | 19.444% | 17.391% | 0.178466 | REVIEW |
| binary_1_tail_average_cash_or_nothing_weighted_basket_1 | Binary | log_linear | 18.580% | 12.625% | 0.100579 | REVIEW |
| barrier_1_upper_knock_in_discrete_cash_weighted_basket_1 | Barrier | linear | 18.007% | 14.660% | 0.046760 | REVIEW |
| binary_1_tail_average_asset_or_nothing_weighted_basket_1 | Binary | log_linear | 17.969% | 12.287% | 0.101055 | REVIEW |
| barrier_1_lower_knock_in_discrete_cash_weighted_basket_1 | Barrier | nearest | 14.976% | 14.660% | 0.172355 | REVIEW |
| barrier_1_lower_knock_in_discrete_call_weighted_basket_1 | Barrier | log_linear | 11.087% | 11.051% | 0.000659 | REVIEW |
| barrier_1_lower_knock_in_discrete_put_weighted_basket_1 | Barrier | log_linear | 9.765% | 8.470% | 0.011329 | REVIEW |
| barrier_1_upper_knock_in_discrete_put_weighted_basket_1 | Barrier | log_linear | 9.421% | 8.853% | 0.000611 | REVIEW |
| barrier_1_lower_knock_in_continuous_put_weighted_basket_1 | Barrier | linear | 9.009% | 8.251% | 0.010624 | REVIEW |
| lookback_1_maximum_floating_strike | Lookback | log_linear | 8.877% | 8.856% | 0.004463 | REVIEW |

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
