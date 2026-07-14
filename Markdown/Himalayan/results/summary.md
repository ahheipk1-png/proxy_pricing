# Himalayan Family Proxy Study

This is the family-level split of the generic exotic payoff pipeline:

```text
Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff
```

Unlike the aggregate SingleExotic/BasketExotic smoke studies, this file
contains at least 100 single-underlying and 100 basket configurations for `Himalayan`.

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
- elapsed seconds: `328.1`

## Accuracy Summary

- PASS: `200`
- WATCH: `0`
- REVIEW: `0`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 0.641% | 0.163% | 0.367% | 0.000457 | 100 | 0 | 0 |
| basket | 100 | 0.329% | 0.082% | 0.217% | 0.000222 | 100 | 0 | 0 |

## Subtype Coverage

- observations: `arithmetic_average, geometric_average, spot, tail_average`
- performances: `fixed_notional`
- rankings: `order_statistic, weighted_basket`
- aggregations: `average, compounded, maximum, sum`
- payoffs: `clamped_linear, option`
- proxy methods selected: `log_linear, log_pchip, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| himalayan_1_p6_tail_average_rank1_remove0_sum_wide_call0 | single | log_pchip | 0.641% | 0.408% | 0.000289 | PASS |
| himalayan_1_p4_arithmetic_average_rank0_remove0_compounded_tight_call0 | single | pchip | 0.624% | 0.464% | 0.000298 | PASS |
| himalayan_1_p4_arithmetic_average_rank1_remove0_compounded_tight_cap35 | single | pchip | 0.622% | 0.468% | 0.000228 | PASS |
| himalayan_1_p6_arithmetic_average_rank1_remove0_compounded_tight_cap35 | single | pchip | 0.469% | 0.366% | 0.000394 | PASS |
| himalayan_1_p6_spot_rank0_remove0_sum_wide_cap35 | single | log_pchip | 0.457% | 0.376% | 0.000667 | PASS |
| himalayan_1_p6_arithmetic_average_rank0_remove0_compounded_tight_call0 | single | pchip | 0.451% | 0.343% | 0.000401 | PASS |
| himalayan_1_p3_spot_rank0_remove0_sum_wide_cap35 | single | log_pchip | 0.448% | 0.296% | 0.000198 | PASS |
| himalayan_1_p4_tail_average_rank1_remove0_sum_wide_call0 | single | log_pchip | 0.424% | 0.349% | 0.000210 | PASS |
| himalayan_1_p2_spot_rank1_remove0_compounded_wide_cap50 | single | log_linear | 0.412% | 0.393% | 0.003027 | PASS |
| himalayan_1_p6_geometric_average_rank0_remove0_sum_low_cap_call0 | single | log_pchip | 0.363% | 0.276% | 0.000329 | PASS |
| himalayan_1_p6_geometric_average_rank1_remove0_sum_low_cap_cap35 | single | log_pchip | 0.363% | 0.276% | 0.000297 | PASS |
| himalayan_1_p4_spot_rank0_remove0_sum_wide_cap35 | single | log_pchip | 0.351% | 0.230% | 0.000324 | PASS |
| himalayan_1_p4_spot_rank1_remove0_sum_tight_cap50 | single | pchip | 0.340% | 0.262% | 0.000221 | PASS |
| himalayan_4_p2_spot_rank4_remove0_compounded_wide_cap50 | basket | pchip | 0.329% | 0.276% | 0.000104 | PASS |
| himalayan_4_p4_geometric_average_rank0_remove0_maximum_low_cap_cap50 | basket | pchip | 0.323% | 0.258% | 0.000106 | PASS |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\HimalayanOptExperiment\results\himalayan_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\HimalayanOptExperiment\results\himalayan_family_proxy_details.csv`
