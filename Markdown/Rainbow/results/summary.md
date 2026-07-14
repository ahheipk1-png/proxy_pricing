# Rainbow Family Proxy Study

This is the family-level split of the generic exotic payoff pipeline:

```text
Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff
```

Unlike the aggregate SingleExotic/BasketExotic smoke studies, this file
contains at least 100 single-underlying and 100 basket configurations for `Rainbow`.

## Setup

- total configurations priced: `200`
- single configurations: `100`
- basket configurations: `100`
- train states per configuration: `121` common spot-scale states
- validation states per configuration: `61` shifted spot-scale states
- train paths per state label: `16,384` low-discrepancy antithetic paths
- benchmark paths per validation state: `16,384` independent low-discrepancy antithetic paths
- validation uses a separate path-ratio stream at shifted scale states
  so the reported error includes out-of-sample Monte Carlo benchmark noise
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `187.8`

## Accuracy Summary

- PASS: `91`
- WATCH: `17`
- REVIEW: `92`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 23.467% | 7.234% | 19.398% | 0.007645 | 54 | 6 | 40 |
| basket | 100 | 28.638% | 8.221% | 22.593% | 0.024674 | 37 | 11 | 52 |

## Subtype Coverage

- observations: `arithmetic_average, geometric_average, spot, tail_average, weighted_average`
- performances: `fixed_notional, fixed_unit, relative, spot_ratio`
- rankings: `order_statistic, weighted_basket`
- aggregations: `sum`
- payoffs: `option`
- proxy methods selected: `linear, log_linear, log_pchip, logit_linear, logit_pchip, nearest, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| rainbow_4_avg_back_weighted_notional_105_raw_worst_call0 | basket | log_linear | 28.638% | 27.602% | 0.022313 | REVIEW |
| rainbow_4_avg_full_unit_95_floor20_cap35_worst_call0 | basket | log_linear | 27.644% | 25.581% | 0.016473 | REVIEW |
| rainbow_4_tail_6_notional_95_raw_worst_call0 | basket | log_linear | 25.729% | 25.207% | 0.046083 | REVIEW |
| rainbow_4_avg_back_unit_100_raw_worst_call0 | basket | log_linear | 25.709% | 25.344% | 0.041624 | REVIEW |
| rainbow_4_tail_6_unit_100_floor10_cap20_worst_call0 | basket | log_linear | 25.688% | 25.126% | 0.026063 | REVIEW |
| rainbow_4_geo_full_notional_105_floor30_worst_call0 | basket | log_linear | 24.744% | 22.459% | 0.008149 | REVIEW |
| rainbow_1_spot_t12_relative_floor20_cap35_weighted_call5 | single | linear | 23.467% | 22.167% | 0.014670 | REVIEW |
| rainbow_1_spot_t12_notional_105_zero_cap30_weighted_call0 | single | linear | 23.428% | 21.552% | 0.013460 | REVIEW |
| rainbow_1_spot_t12_notional_95_floor30_weighted_call5 | single | linear | 23.363% | 22.233% | 0.011934 | REVIEW |
| rainbow_1_spot_t12_notional_95_raw_weighted_call0 | single | linear | 23.303% | 21.451% | 0.011694 | REVIEW |
| rainbow_4_avg_quarter_notional_95_floor10_cap20_worst_call0 | basket | log_linear | 22.105% | 20.919% | 0.012111 | REVIEW |
| rainbow_1_spot_t12_unit_100_floor10_cap20_weighted_call0 | single | linear | 21.690% | 20.081% | 0.015244 | REVIEW |
| rainbow_1_spot_t12_unit_95_floor10_cap20_weighted_call5 | single | linear | 20.409% | 19.362% | 0.013811 | REVIEW |
| rainbow_1_avg_front_weighted_unit_100_floor30_weighted_call5 | single | log_linear | 19.457% | 18.091% | 0.003128 | REVIEW |
| rainbow_1_avg_front_weighted_unit_95_raw_weighted_call5 | single | nearest | 19.397% | 19.199% | 0.148434 | REVIEW |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\RainbowOptExperiment\results\rainbow_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\RainbowOptExperiment\results\rainbow_family_proxy_details.csv`
