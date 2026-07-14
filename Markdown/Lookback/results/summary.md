# Lookback Family Proxy Study

This is the family-level split of the generic exotic payoff pipeline:

```text
Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff
```

Unlike the aggregate SingleExotic/BasketExotic smoke studies, this file
contains at least 100 single-underlying and 100 basket configurations for `Lookback`.

## Setup

- total configurations priced: `200`
- single configurations: `100`
- basket configurations: `100`
- train states per configuration: `183` common spot-scale states
- validation states per configuration: `61` shifted spot-scale states
- path ratios per state label: `16,384` low-discrepancy antithetic paths
- validation reuses the same Sobol path-ratio stream at shifted scale states
  to isolate proxy interpolation error from Monte Carlo sampling noise
- proxy candidates: direct/log linear interpolation and direct/log PCHIP interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `246.9`

## Accuracy Summary

- PASS: `200`
- WATCH: `0`
- REVIEW: `0`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 2.347% | 0.062% | 0.153% | 0.000101 | 100 | 0 | 0 |
| basket | 100 | 2.434% | 0.067% | 0.119% | 0.000100 | 100 | 0 | 0 |

## Subtype Coverage

- observations: `lookback`
- performances: `relative`
- rankings: `weighted_basket`
- aggregations: `maximum`
- payoffs: `option`
- proxy methods selected: `linear, log_linear, log_pchip, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| lookback_4_full_minimum_fixed_strike_tail3_uncapped_1.1_call | basket | linear | 2.434% | 1.133% | 0.000505 | PASS |
| lookback_1_full_minimum_fixed_strike_tail3_uncapped_1.1_call | single | linear | 2.347% | 1.110% | 0.000512 | PASS |
| lookback_4_quarterly_minimum_fixed_strike_tail3_uncapped_1.1_call | basket | linear | 1.421% | 0.685% | 0.000467 | PASS |
| lookback_1_quarterly_minimum_fixed_strike_tail3_uncapped_1.1_call | single | linear | 1.334% | 0.648% | 0.000468 | PASS |
| lookback_4_full_minimum_fixed_strike_tail1_normal_0.9_call | basket | linear | 0.621% | 0.471% | 0.000659 | PASS |
| lookback_1_full_minimum_fixed_strike_tail1_normal_0.9_call | single | linear | 0.607% | 0.440% | 0.000642 | PASS |
| lookback_4_quarterly_minimum_fixed_strike_tail1_normal_0.9_call | basket | linear | 0.426% | 0.367% | 0.000616 | PASS |
| lookback_1_quarterly_minimum_fixed_strike_tail1_normal_0.9_call | single | linear | 0.385% | 0.324% | 0.000600 | PASS |
| lookback_1_quarterly_maximum_fixed_strike_tail3_wide_1.1_call | single | pchip | 0.224% | 0.150% | 0.000251 | PASS |
| lookback_1_after_q1_minimum_fixed_strike_tail1_normal_0.9_call | single | log_pchip | 0.217% | 0.152% | 0.000101 | PASS |
| lookback_1_quarterly_trimmed_average_floating_ratio_tail3_normal_0.9_put | single | log_linear | 0.175% | 0.172% | 0.000014 | PASS |
| lookback_1_quarterly_average_fixed_strike_tail6_uncapped_1.0_put | single | pchip | 0.172% | 0.127% | 0.000083 | PASS |
| lookback_4_after_q1_minimum_fixed_strike_tail3_uncapped_1.1_call | basket | log_pchip | 0.166% | 0.162% | 0.000059 | PASS |
| lookback_4_after_q1_minimum_fixed_strike_tail1_normal_0.9_call | basket | log_pchip | 0.161% | 0.116% | 0.000109 | PASS |
| lookback_4_quarterly_maximum_fixed_strike_tail3_wide_1.1_call | basket | log_pchip | 0.146% | 0.107% | 0.000287 | PASS |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\LookbackOptExperiment\results\lookback_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\LookbackOptExperiment\results\lookback_family_proxy_details.csv`
