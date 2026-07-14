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
- benchmark paths per validation state: `65,536` independent low-discrepancy antithetic paths
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `1345.2`

## Accuracy Summary

- PASS: `79`
- WATCH: `19`
- REVIEW: `22`

| Family | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Barrier | 23 | 13.916% | 3.628% | 11.512% | 0.024213 | 16 | 5 | 2 |
| Binary | 23 | 56.590% | 21.272% | 52.807% | 0.089890 | 3 | 0 | 20 |
| Himalayan | 18 | 5.752% | 3.247% | 5.190% | 0.009080 | 16 | 2 | 0 |
| Lookback | 16 | 8.308% | 1.789% | 6.976% | 0.008162 | 13 | 3 | 0 |
| Rainbow | 22 | 7.625% | 4.159% | 6.899% | 0.009175 | 13 | 9 | 0 |
| YieldSeeker | 18 | 0.043% | 0.020% | 0.033% | 0.003966 | 18 | 0 | 0 |

## Worst Cases

| Case | Family | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| binary_4_spot_asset_or_nothing_weighted_basket_1 | Binary | nearest | 56.590% | 54.225% | 0.261131 | REVIEW |
| binary_4_spot_cash_or_nothing_weighted_basket_1 | Binary | nearest | 55.710% | 54.545% | 0.177900 | REVIEW |
| binary_4_arithmetic_average_asset_or_nothing_order_statistic_4 | Binary | log_linear | 40.660% | 40.040% | 0.056332 | REVIEW |
| binary_4_arithmetic_average_cash_or_nothing_order_statistic_4 | Binary | log_linear | 40.503% | 39.397% | 0.055961 | REVIEW |
| binary_4_arithmetic_average_double_digital_order_statistic_4 | Binary | nearest | 34.146% | 32.253% | 0.247603 | REVIEW |
| binary_4_arithmetic_average_range_digital_order_statistic_4 | Binary | nearest | 34.146% | 32.253% | 0.247603 | REVIEW |
| binary_4_spot_asset_or_nothing_order_statistic_4 | Binary | nearest | 31.107% | 25.689% | 0.204748 | REVIEW |
| binary_4_spot_cash_or_nothing_order_statistic_4 | Binary | linear | 29.321% | 23.323% | 0.090573 | REVIEW |
| binary_4_spot_double_digital_order_statistic_4 | Binary | linear | 28.255% | 23.083% | 0.108062 | REVIEW |
| binary_4_spot_range_digital_order_statistic_4 | Binary | linear | 28.255% | 23.083% | 0.108062 | REVIEW |
| binary_4_arithmetic_average_double_digital_weighted_basket_1 | Binary | log_pchip | 20.085% | 16.066% | 0.049671 | REVIEW |
| binary_4_arithmetic_average_range_digital_weighted_basket_1 | Binary | log_pchip | 20.085% | 16.066% | 0.049671 | REVIEW |
| binary_4_tail_average_cash_or_nothing_weighted_basket_1 | Binary | linear | 18.562% | 16.845% | 0.044233 | REVIEW |
| binary_4_tail_average_asset_or_nothing_weighted_basket_1 | Binary | linear | 18.322% | 17.421% | 0.044751 | REVIEW |
| binary_4_spot_double_digital_weighted_basket_1 | Binary | log_pchip | 16.973% | 14.011% | 0.077317 | REVIEW |

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
