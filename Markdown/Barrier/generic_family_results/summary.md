# Barrier Family Proxy Study

This is the family-level split of the generic exotic payoff pipeline:

```text
Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff
```

Unlike the aggregate SingleExotic/BasketExotic smoke studies, this file
contains at least 100 single-underlying and 100 basket configurations for `Barrier`.

## Setup

- total configurations priced: `200`
- single configurations: `100`
- basket configurations: `100`
- train states per configuration: `300` common spot-scale states
- validation states per configuration: `61` shifted spot-scale states
- path ratios per state label: `32,768` low-discrepancy antithetic paths
- validation reuses the same Sobol path-ratio stream at shifted scale states
  to isolate proxy interpolation error from Monte Carlo sampling noise
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `1440.6`

## Accuracy Summary

- PASS: `190`
- WATCH: `10`
- REVIEW: `0`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 5.864% | 1.294% | 5.316% | 0.009457 | 92 | 8 | 0 |
| basket | 100 | 5.858% | 1.107% | 3.282% | 0.003544 | 98 | 2 | 0 |

## Subtype Coverage

- observations: `spot`
- performances: `relative`
- rankings: `order_statistic, weighted_basket`
- aggregations: `sum`
- payoffs: `option`
- proxy methods selected: `linear, log_linear, log_pchip, logit_linear, logit_pchip, nearest, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| barrier_1_upper_0.15_knock_out_discrete_monthly_cash_0.05_0.0_weighted | single | nearest | 5.864% | 5.316% | 0.157612 | WATCH |
| barrier_1_upper_0.15_knock_out_continuous_quarterly_cash_0.0_0.0_weighted | single | nearest | 5.864% | 5.316% | 0.157612 | WATCH |
| barrier_1_upper_0.15_knock_out_continuous_semiannual_cash_-0.05_0.0_weighted | single | nearest | 5.864% | 5.316% | 0.157612 | WATCH |
| barrier_1_upper_0.15_knock_out_continuous_late_cash_-0.05_0.0_weighted | single | nearest | 5.864% | 5.316% | 0.157612 | WATCH |
| barrier_4_lower_-0.35_knock_in_discrete_semiannual_put_0.05_0.0_weighted | basket | linear | 5.858% | 5.763% | 0.002420 | WATCH |
| barrier_1_upper_0.25_knock_in_discrete_monthly_cash_0.05_0.0_weighted | single | logit_pchip | 5.419% | 5.322% | 0.006694 | WATCH |
| barrier_1_upper_0.25_knock_in_continuous_quarterly_cash_0.0_0.0_weighted | single | logit_pchip | 5.419% | 5.322% | 0.006694 | WATCH |
| barrier_1_upper_0.25_knock_in_continuous_semiannual_cash_-0.05_0.0_weighted | single | logit_pchip | 5.419% | 5.322% | 0.006694 | WATCH |
| barrier_1_upper_0.25_knock_in_continuous_late_cash_-0.05_0.0_weighted | single | logit_pchip | 5.419% | 5.322% | 0.006694 | WATCH |
| barrier_4_upper_0.25_knock_in_continuous_late_cash_-0.05_0.0_weighted | basket | logit_pchip | 5.375% | 5.305% | 0.007036 | WATCH |
| barrier_4_lower_-0.25_knock_in_continuous_quarterly_cash_0.0_0.0_best | basket | pchip | 5.224% | 4.963% | 0.008082 | PASS |
| barrier_4_upper_0.25_knock_in_continuous_semiannual_cash_-0.05_0.0_worst | basket | pchip | 4.299% | 3.643% | 0.007216 | PASS |
| barrier_4_upper_0.25_knock_out_continuous_monthly_call_0.0_0.0_weighted | basket | linear | 3.795% | 3.265% | 0.001344 | PASS |
| barrier_4_upper_0.15_knock_out_continuous_quarterly_cash_0.0_0.0_best | basket | linear | 3.786% | 3.613% | 0.009286 | PASS |
| barrier_4_lower_-0.15_knock_in_continuous_monthly_call_0.0_0.0_weighted | basket | logit_pchip | 3.595% | 2.622% | 0.000363 | PASS |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\BarrierFamilyOptExperiment\results\barrier_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\BarrierFamilyOptExperiment\results\barrier_family_proxy_details.csv`
