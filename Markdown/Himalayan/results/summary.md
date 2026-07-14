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
- train paths per state label: `16,384` low-discrepancy antithetic paths
- benchmark paths per validation state: `16,384` independent low-discrepancy antithetic paths
- validation uses a separate path-ratio stream at shifted scale states
  so the reported error includes out-of-sample Monte Carlo benchmark noise
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `322.4`

## Accuracy Summary

- PASS: `58`
- WATCH: `36`
- REVIEW: `106`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 21.582% | 10.943% | 18.773% | 0.016367 | 14 | 12 | 74 |
| basket | 100 | 52.732% | 8.109% | 20.109% | 0.027641 | 44 | 24 | 32 |

## Subtype Coverage

- observations: `arithmetic_average, geometric_average, spot, tail_average`
- performances: `fixed_notional`
- rankings: `order_statistic, weighted_basket`
- aggregations: `average, compounded, maximum, sum`
- payoffs: `clamped_linear, option`
- proxy methods selected: `linear, log_linear, log_pchip, logit_linear, logit_pchip, nearest, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| himalayan_4_p4_spot_rank4_remove0_compounded_wide_cap50 | basket | log_linear | 52.732% | 47.789% | 0.041217 | REVIEW |
| himalayan_4_p6_spot_rank4_remove0_compounded_wide_cap50 | basket | log_linear | 46.870% | 42.654% | 0.039668 | REVIEW |
| himalayan_4_p3_spot_rank4_remove0_compounded_wide_cap50 | basket | log_linear | 35.960% | 33.268% | 0.040858 | REVIEW |
| himalayan_4_p6_tail_average_rank0_remove0_sum_low_cap_call0 | basket | logit_pchip | 31.840% | 31.633% | 0.067748 | REVIEW |
| himalayan_4_p4_tail_average_rank0_remove0_sum_low_cap_call0 | basket | pchip | 25.628% | 24.930% | 0.042820 | REVIEW |
| himalayan_1_p2_spot_rank1_remove0_maximum_positive_call0 | single | linear | 21.582% | 20.037% | 0.013248 | REVIEW |
| himalayan_4_p2_spot_rank4_remove0_compounded_wide_cap50 | basket | log_linear | 21.192% | 19.856% | 0.035873 | REVIEW |
| himalayan_1_p3_spot_rank1_remove0_maximum_positive_call0 | single | linear | 20.925% | 19.301% | 0.010321 | REVIEW |
| himalayan_1_p2_spot_rank0_remove0_compounded_positive_cap35 | single | linear | 20.729% | 19.337% | 0.023277 | REVIEW |
| himalayan_1_p2_arithmetic_average_rank0_remove0_compounded_tight_call0 | single | log_linear | 20.321% | 19.324% | 0.016688 | REVIEW |
| himalayan_1_p2_arithmetic_average_rank1_remove0_compounded_tight_cap35 | single | log_linear | 20.321% | 19.324% | 0.016688 | REVIEW |
| himalayan_1_p6_spot_rank0_remove0_compounded_positive_cap35 | single | linear | 19.542% | 18.745% | 0.021923 | REVIEW |
| himalayan_1_p6_arithmetic_average_rank1_remove0_compounded_tight_cap35 | single | log_linear | 18.559% | 17.784% | 0.019125 | REVIEW |
| himalayan_1_p6_arithmetic_average_rank0_remove0_compounded_tight_call0 | single | log_linear | 18.394% | 18.300% | 0.059067 | REVIEW |
| himalayan_4_p2_arithmetic_average_rank4_remove0_average_wide_call0 | basket | log_linear | 18.269% | 18.135% | 0.013222 | REVIEW |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\HimalayanOptExperiment\results\himalayan_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\HimalayanOptExperiment\results\himalayan_family_proxy_details.csv`
