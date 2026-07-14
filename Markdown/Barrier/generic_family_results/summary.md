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
- train paths per state label: `32,768` low-discrepancy antithetic paths
- benchmark paths per validation state: `32,768` independent low-discrepancy antithetic paths
- validation uses a separate path-ratio stream at shifted scale states
  so the reported error includes out-of-sample Monte Carlo benchmark noise
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `1016.0`

## Accuracy Summary

- PASS: `80`
- WATCH: `25`
- REVIEW: `95`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 43.243% | 13.348% | 38.576% | 0.035276 | 38 | 8 | 54 |
| basket | 100 | 70.370% | 9.016% | 25.821% | 0.049775 | 42 | 17 | 41 |

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
| barrier_4_lower_-0.25_knock_in_continuous_quarterly_cash_0.0_0.0_best | basket | nearest | 70.370% | 51.225% | 0.144057 | REVIEW |
| barrier_4_upper_0.25_knock_in_continuous_semiannual_cash_-0.05_0.0_worst | basket | log_pchip | 54.895% | 47.647% | 0.096681 | REVIEW |
| barrier_1_lower_-0.25_knock_in_discrete_monthly_cash_0.05_0.0_weighted | single | logit_pchip | 43.243% | 38.576% | 0.061874 | REVIEW |
| barrier_1_lower_-0.25_knock_in_continuous_quarterly_cash_0.0_0.0_weighted | single | logit_pchip | 43.243% | 38.576% | 0.061874 | REVIEW |
| barrier_1_lower_-0.25_knock_in_continuous_semiannual_cash_-0.05_0.0_weighted | single | logit_pchip | 43.243% | 38.576% | 0.061874 | REVIEW |
| barrier_1_lower_-0.25_knock_in_continuous_late_cash_-0.05_0.0_weighted | single | logit_pchip | 43.243% | 38.576% | 0.061874 | REVIEW |
| barrier_1_upper_0.25_knock_in_discrete_monthly_cash_0.05_0.0_weighted | single | linear | 42.467% | 41.007% | 0.059819 | REVIEW |
| barrier_1_upper_0.25_knock_in_continuous_quarterly_cash_0.0_0.0_weighted | single | linear | 42.467% | 41.007% | 0.059819 | REVIEW |
| barrier_1_upper_0.25_knock_in_continuous_semiannual_cash_-0.05_0.0_weighted | single | linear | 42.467% | 41.007% | 0.059819 | REVIEW |
| barrier_1_upper_0.25_knock_in_continuous_late_cash_-0.05_0.0_weighted | single | linear | 42.467% | 41.007% | 0.059819 | REVIEW |
| barrier_1_lower_-0.35_knock_in_discrete_monthly_put_0.05_0.0_weighted | single | nearest | 40.565% | 34.403% | 0.055478 | REVIEW |
| barrier_1_upper_0.35_knock_in_continuous_late_call_-0.05_0.0_weighted | single | linear | 35.601% | 31.987% | 0.023867 | REVIEW |
| barrier_1_lower_-0.35_knock_in_continuous_quarterly_put_0.0_0.0_weighted | single | nearest | 35.288% | 32.300% | 0.049272 | REVIEW |
| barrier_4_upper_0.15_knock_out_continuous_late_cash_-0.05_0.0_weighted | basket | linear | 33.386% | 29.563% | 0.080227 | REVIEW |
| barrier_1_upper_0.35_knock_in_continuous_quarterly_call_0.0_0.0_weighted | single | linear | 30.910% | 30.018% | 0.021126 | REVIEW |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\BarrierFamilyOptExperiment\results\barrier_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\BarrierFamilyOptExperiment\results\barrier_family_proxy_details.csv`
