# Binary Family Proxy Study

This is the family-level split of the generic exotic payoff pipeline:

```text
Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff
```

Unlike the aggregate SingleExotic/BasketExotic smoke studies, this file
contains at least 100 single-underlying and 100 basket configurations for `Binary`.

## Setup

- total configurations priced: `200`
- single configurations: `100`
- basket configurations: `100`
- train states per configuration: `183` common spot-scale states
- validation states per configuration: `61` shifted spot-scale states
- path ratios per state label: `131,072` low-discrepancy antithetic paths
- validation reuses the same Sobol path-ratio stream at shifted scale states
  to isolate proxy interpolation error from Monte Carlo sampling noise
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `1522.3`

## Accuracy Summary

- PASS: `199`
- WATCH: `1`
- REVIEW: `0`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 7.270% | 1.642% | 3.675% | 0.007062 | 99 | 1 | 0 |
| basket | 100 | 5.772% | 1.693% | 3.833% | 0.007676 | 100 | 0 | 0 |

## Subtype Coverage

- observations: `arithmetic_average, geometric_average, spot, tail_average, weighted_average`
- performances: `fixed_notional, fixed_unit, relative`
- rankings: `order_statistic, weighted_basket`
- aggregations: `sum`
- payoffs: `linear`
- proxy methods selected: `linear, log_linear, log_pchip, logit_linear, logit_pchip, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| binary_1_avg_quarter_unit_100_cash_or_nothing_0.15_0.0_0.25_up_weighted | single | log_linear | 7.270% | 4.645% | 0.006545 | PASS |
| binary_4_spot_t6_relative_double_digital_-0.15_0.0_0.25_up_middle | basket | log_linear | 5.772% | 4.054% | 0.011353 | PASS |
| binary_1_avg_quarter_notional_100_range_digital_-0.08_-0.08_0.18_down_weighted | single | pchip | 5.543% | 5.473% | 0.009257 | WATCH |
| binary_1_geo_full_relative_double_digital_0.15_-0.08_0.18_down_weighted | single | linear | 5.245% | 3.631% | 0.012666 | PASS |
| binary_4_geo_full_relative_double_digital_0.15_-0.08_0.18_down_weighted | basket | linear | 4.930% | 4.570% | 0.013689 | PASS |
| binary_4_avg_front_unit_100_double_digital_-0.08_0.0_0.25_up_middle | basket | linear | 4.823% | 3.211% | 0.012291 | PASS |
| binary_4_spot_t6_unit_100_asset_or_nothing_0.0_-0.15_0.15_up_weighted | basket | log_linear | 4.534% | 3.964% | 0.006320 | PASS |
| binary_4_avg_front_unit_100_asset_or_nothing_0.08_-0.08_0.18_down_weighted | basket | linear | 4.478% | 2.993% | 0.008464 | PASS |
| binary_4_avg_front_weighted_relative_cash_or_nothing_0.15_-0.08_0.18_down_weighted | basket | linear | 4.375% | 3.011% | 0.006891 | PASS |
| binary_4_avg_front_weighted_relative_range_digital_-0.08_-0.15_0.15_up_weighted | basket | linear | 4.375% | 3.638% | 0.016806 | PASS |
| binary_4_tail_6_relative_cash_or_nothing_-0.08_-0.08_0.18_down_weighted | basket | logit_linear | 4.300% | 4.083% | 0.005569 | PASS |
| binary_4_geo_full_notional_100_double_digital_0.15_-0.15_0.15_down_middle | basket | pchip | 4.198% | 3.740% | 0.012883 | PASS |
| binary_1_avg_back_weighted_unit_100_double_digital_0.15_0.0_0.25_up_weighted | single | logit_pchip | 4.176% | 3.883% | 0.011265 | PASS |
| binary_4_geo_full_unit_100_double_digital_0.15_-0.15_0.15_up_weighted | basket | logit_linear | 4.163% | 4.014% | 0.009820 | PASS |
| binary_4_avg_front_relative_cash_or_nothing_-0.15_0.0_0.25_up_middle | basket | linear | 3.966% | 3.684% | 0.006709 | PASS |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\BinaryOptExperiment\results\binary_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\BinaryOptExperiment\results\binary_family_proxy_details.csv`
