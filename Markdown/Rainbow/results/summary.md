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
- path ratios per state label: `16,384` low-discrepancy antithetic paths
- validation reuses the same Sobol path-ratio stream at shifted scale states
  to isolate proxy interpolation error from Monte Carlo sampling noise
- proxy candidates: direct/log linear interpolation and direct/log PCHIP interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `113.4`

## Accuracy Summary

- PASS: `200`
- WATCH: `0`
- REVIEW: `0`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 0.322% | 0.078% | 0.174% | 0.000080 | 100 | 0 | 0 |
| basket | 100 | 0.246% | 0.070% | 0.177% | 0.000094 | 100 | 0 | 0 |

## Subtype Coverage

- observations: `arithmetic_average, geometric_average, spot, tail_average, weighted_average`
- performances: `fixed_notional, fixed_unit, relative, spot_ratio`
- rankings: `order_statistic, weighted_basket`
- aggregations: `sum`
- payoffs: `option`
- proxy methods selected: `linear, log_linear, log_pchip, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| rainbow_1_geo_full_notional_95_floor20_cap35_weighted_call5 | single | pchip | 0.322% | 0.224% | 0.000088 | PASS |
| rainbow_1_avg_front_weighted_unit_100_floor30_weighted_call5 | single | pchip | 0.292% | 0.205% | 0.000060 | PASS |
| rainbow_1_avg_front_weighted_notional_105_floor20_cap35_weighted_call0 | single | pchip | 0.278% | 0.199% | 0.000089 | PASS |
| rainbow_4_avg_back_weighted_notional_105_raw_worst_call0 | basket | log_pchip | 0.246% | 0.176% | 0.000034 | PASS |
| rainbow_4_geo_full_unit_100_zero_cap30_middle_call5 | basket | pchip | 0.246% | 0.210% | 0.000104 | PASS |
| rainbow_4_avg_front_weighted_notional_100_floor20_cap35_worst_call0 | basket | log_pchip | 0.238% | 0.166% | 0.000074 | PASS |
| rainbow_4_geo_full_notional_100_floor20_cap35_weighted_call0 | basket | log_pchip | 0.238% | 0.197% | 0.000049 | PASS |
| rainbow_4_avg_full_unit_95_floor20_cap35_worst_call0 | basket | pchip | 0.227% | 0.188% | 0.000094 | PASS |
| rainbow_4_avg_quarter_notional_95_floor10_cap20_worst_call0 | basket | log_pchip | 0.219% | 0.177% | 0.000060 | PASS |
| rainbow_4_spot_t6_unit_95_floor10_cap20_weighted_put0 | basket | log_pchip | 0.211% | 0.188% | 0.000070 | PASS |
| rainbow_1_avg_quarter_relative_floor30_weighted_call0 | single | log_pchip | 0.204% | 0.105% | 0.000048 | PASS |
| rainbow_1_geo_full_notional_100_floor20_cap35_weighted_call0 | single | pchip | 0.197% | 0.180% | 0.000080 | PASS |
| rainbow_1_geo_full_relative_zero_cap30_weighted_call0 | single | pchip | 0.197% | 0.180% | 0.000080 | PASS |
| rainbow_1_avg_quarter_notional_100_floor10_cap20_weighted_call0 | single | log_pchip | 0.196% | 0.121% | 0.000066 | PASS |
| rainbow_4_geo_full_notional_105_floor30_worst_call0 | basket | pchip | 0.195% | 0.186% | 0.000052 | PASS |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\RainbowOptExperiment\results\rainbow_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\RainbowOptExperiment\results\rainbow_family_proxy_details.csv`
