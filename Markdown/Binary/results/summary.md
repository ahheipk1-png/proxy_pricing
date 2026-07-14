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
- train paths per state label: `131,072` low-discrepancy antithetic paths
- benchmark paths per validation state: `131,072` independent low-discrepancy antithetic paths
- validation uses a separate path-ratio stream at shifted scale states
  so the reported error includes out-of-sample Monte Carlo benchmark noise
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `1185.8`

## Accuracy Summary

- PASS: `52`
- WATCH: `34`
- REVIEW: `114`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 31.090% | 11.127% | 23.723% | 0.043628 | 26 | 21 | 53 |
| basket | 100 | 39.927% | 12.327% | 28.223% | 0.052140 | 26 | 13 | 61 |

## Subtype Coverage

- observations: `arithmetic_average, geometric_average, spot, tail_average, weighted_average`
- performances: `fixed_notional, fixed_unit, relative`
- rankings: `order_statistic, weighted_basket`
- aggregations: `sum`
- payoffs: `linear`
- proxy methods selected: `linear, log_linear, log_pchip, nearest, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| binary_4_avg_quarter_relative_asset_or_nothing_-0.15_-0.15_0.15_down_middle | basket | pchip | 39.927% | 34.769% | 0.037455 | REVIEW |
| binary_4_tail_6_notional_100_cash_or_nothing_-0.08_-0.15_0.15_down_middle | basket | nearest | 38.356% | 31.604% | 0.209417 | REVIEW |
| binary_4_avg_full_relative_range_digital_0.08_-0.15_0.15_down_middle | basket | linear | 36.455% | 30.874% | 0.049600 | REVIEW |
| binary_4_avg_back_weighted_unit_100_cash_or_nothing_0.08_-0.15_0.15_down_middle | basket | pchip | 35.482% | 33.582% | 0.043725 | REVIEW |
| binary_1_avg_front_relative_asset_or_nothing_0.15_-0.15_0.15_up_weighted | single | nearest | 31.090% | 24.776% | 0.282732 | REVIEW |
| binary_4_avg_full_unit_100_range_digital_0.0_0.0_0.25_up_middle | basket | linear | 31.019% | 30.696% | 0.049550 | REVIEW |
| binary_1_avg_quarter_relative_range_digital_-0.08_0.0_0.25_up_weighted | single | linear | 30.379% | 27.060% | 0.038686 | REVIEW |
| binary_4_geo_full_notional_100_cash_or_nothing_0.0_0.0_0.25_up_middle | basket | linear | 29.995% | 28.093% | 0.031856 | REVIEW |
| binary_1_avg_quarter_unit_100_cash_or_nothing_0.15_0.0_0.25_up_weighted | single | pchip | 28.688% | 25.239% | 0.021590 | REVIEW |
| binary_1_avg_quarter_notional_100_range_digital_-0.08_-0.08_0.18_down_weighted | single | linear | 28.627% | 27.989% | 0.039868 | REVIEW |
| binary_1_avg_full_relative_asset_or_nothing_-0.08_0.0_0.25_up_weighted | single | linear | 28.079% | 20.271% | 0.017518 | REVIEW |
| binary_4_avg_front_weighted_notional_100_cash_or_nothing_0.15_-0.15_0.15_down_middle | basket | linear | 27.615% | 21.354% | 0.034409 | REVIEW |
| binary_4_avg_full_relative_double_digital_-0.15_-0.15_0.15_up_weighted | basket | linear | 27.567% | 20.478% | 0.032839 | REVIEW |
| binary_4_avg_full_unit_100_asset_or_nothing_-0.08_-0.15_0.15_down_middle | basket | linear | 26.994% | 16.531% | 0.026706 | REVIEW |
| binary_4_avg_quarter_unit_100_cash_or_nothing_0.15_0.0_0.25_up_middle | basket | linear | 26.811% | 20.770% | 0.038627 | REVIEW |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\BinaryOptExperiment\results\binary_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\BinaryOptExperiment\results\binary_family_proxy_details.csv`
