# Basket Generic Exotic Proxy Study

This study implements the requested data-driven product pipeline:

```text
Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff
```

Existing product-specific main scripts were not replaced. This is a new
generic proxy layer for exotic payoff configurations.

## Setup

- configurations priced: `120`
- train states per configuration: `241` common spot-scale states
- validation states per configuration: `101` shifted spot-scale states
- train paths per state label: `65,536` low-discrepancy antithetic paths
- benchmark paths per validation state: `65,536` independent paths
- proxy candidates: direct/log linear interpolation and direct/log PCHIP interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `1783.7`

## Accuracy Summary

- PASS: `116`
- WATCH: `4`
- REVIEW: `0`

| Family | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Barrier | 23 | 2.036% | 0.478% | 1.592% | 0.002425 | 23 | 0 | 0 |
| Binary | 23 | 10.933% | 2.515% | 5.215% | 0.007964 | 19 | 4 | 0 |
| Himalayan | 18 | 0.107% | 0.028% | 0.058% | 0.000035 | 18 | 0 | 0 |
| Lookback | 16 | 7.877% | 0.041% | 0.150% | 0.000076 | 16 | 0 | 0 |
| Rainbow | 22 | 0.050% | 0.019% | 0.029% | 0.000011 | 22 | 0 | 0 |
| YieldSeeker | 18 | 0.008% | 0.002% | 0.005% | 0.000443 | 18 | 0 | 0 |

## Worst Cases

| Case | Family | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| binary_4_arithmetic_average_cash_or_nothing_order_statistic_4 | Binary | log_linear | 10.933% | 5.194% | 0.007042 | WATCH |
| binary_4_arithmetic_average_asset_or_nothing_order_statistic_4 | Binary | log_linear | 10.702% | 5.203% | 0.007057 | WATCH |
| lookback_4_minimum_fixed_strike | Lookback | linear | 7.877% | 0.507% | 0.000392 | PASS |
| binary_4_arithmetic_average_cash_or_nothing_weighted_basket_1 | Binary | linear | 6.464% | 5.216% | 0.006570 | WATCH |
| binary_4_arithmetic_average_asset_or_nothing_weighted_basket_1 | Binary | linear | 6.271% | 5.227% | 0.006518 | WATCH |
| binary_4_arithmetic_average_double_digital_weighted_basket_1 | Binary | log_pchip | 6.260% | 3.630% | 0.012206 | PASS |
| binary_4_arithmetic_average_range_digital_weighted_basket_1 | Binary | log_pchip | 6.260% | 3.630% | 0.012206 | PASS |
| binary_4_tail_average_cash_or_nothing_weighted_basket_1 | Binary | pchip | 3.161% | 2.724% | 0.009414 | PASS |
| binary_4_spot_cash_or_nothing_weighted_basket_1 | Binary | log_linear | 3.085% | 2.187% | 0.007793 | PASS |
| binary_4_tail_average_asset_or_nothing_weighted_basket_1 | Binary | pchip | 3.025% | 2.727% | 0.009416 | PASS |
| binary_4_spot_asset_or_nothing_weighted_basket_1 | Binary | log_linear | 2.945% | 2.087% | 0.007803 | PASS |
| binary_4_spot_asset_or_nothing_order_statistic_4 | Binary | linear | 2.719% | 2.562% | 0.007052 | PASS |
| binary_4_spot_cash_or_nothing_order_statistic_4 | Binary | linear | 2.716% | 2.708% | 0.006959 | PASS |
| binary_4_arithmetic_average_double_digital_order_statistic_4 | Binary | linear | 2.692% | 2.543% | 0.011269 | PASS |
| binary_4_arithmetic_average_range_digital_order_statistic_4 | Binary | linear | 2.692% | 2.543% | 0.011269 | PASS |

## Product Building Blocks Covered

- observations: spot, arithmetic average, tail average, lookback
- performance: fixed notional, fixed unit, relative
- ranking: identity, weighted basket, order statistic
- transformation: floor, cap, combined clamp
- aggregation: average, sum, compounded
- payoff families: rainbow, Himalayan, yield seeker, lookback, barrier, binary

## Files

- case CSV: `C:\codex_proj\proxy_pricing\BasketExoticOptExperiment\results\basket_generic_exotic_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\BasketExoticOptExperiment\results\basket_generic_exotic_proxy_details.csv`
- plot: `C:\codex_proj\proxy_pricing\BasketExoticOptExperiment\results\basket_generic_exotic_proxy_accuracy.png`
