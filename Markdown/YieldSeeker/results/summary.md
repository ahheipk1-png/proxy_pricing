# YieldSeeker Family Proxy Study

This is the family-level split of the generic exotic payoff pipeline:

```text
Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff
```

Unlike the aggregate SingleExotic/BasketExotic smoke studies, this file
contains at least 100 single-underlying and 100 basket configurations for `YieldSeeker`.

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
- elapsed seconds: `126.6`

## Accuracy Summary

- PASS: `200`
- WATCH: `0`
- REVIEW: `0`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 0.071% | 0.011% | 0.035% | 0.002119 | 100 | 0 | 0 |
| basket | 100 | 0.069% | 0.010% | 0.024% | 0.001925 | 100 | 0 | 0 |

## Subtype Coverage

- observations: `spot`
- performances: `relative`
- rankings: `weighted_basket`
- aggregations: `sum`
- payoffs: `linear`
- proxy methods selected: `linear, log_linear, log_pchip, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| yield_seeker_1_memory_True_-0.15_monthly_high | single | log_pchip | 0.071% | 0.055% | 0.009543 | PASS |
| yield_seeker_4_memory_True_-0.15_monthly_high | basket | linear | 0.069% | 0.055% | 0.012122 | PASS |
| yield_seeker_1_memory_True_-0.05_monthly_steep | single | log_linear | 0.065% | 0.063% | 0.015134 | PASS |
| yield_seeker_4_memory_True_0.0_monthly_base | basket | linear | 0.050% | 0.050% | 0.009242 | PASS |
| yield_seeker_1_memory_False_0.05_monthly_defensive | single | log_linear | 0.048% | 0.034% | 0.005244 | PASS |
| yield_seeker_1_memory_True_0.0_monthly_base | single | pchip | 0.048% | 0.047% | 0.008397 | PASS |
| yield_seeker_1_memory_False_0.05_even_months_steep | single | log_pchip | 0.046% | 0.035% | 0.005682 | PASS |
| yield_seeker_4_memory_True_-0.05_monthly_steep | basket | log_pchip | 0.044% | 0.043% | 0.010235 | PASS |
| yield_seeker_1_memory_True_0.15_quarterly_steep | single | log_linear | 0.041% | 0.035% | 0.006113 | PASS |
| yield_seeker_1_memory_False_-0.05_even_months_high | single | log_linear | 0.040% | 0.029% | 0.005761 | PASS |
| yield_seeker_1_memory_False_-0.05_monthly_base | single | log_linear | 0.039% | 0.034% | 0.007244 | PASS |
| yield_seeker_1_memory_True_0.1_monthly_defensive | single | linear | 0.036% | 0.035% | 0.006046 | PASS |
| yield_seeker_4_memory_False_-0.05_monthly_base | basket | linear | 0.035% | 0.031% | 0.008614 | PASS |
| yield_seeker_4_memory_True_0.05_quarterly_high | basket | pchip | 0.033% | 0.023% | 0.004397 | PASS |
| yield_seeker_4_memory_True_0.0_even_months_base | basket | pchip | 0.033% | 0.023% | 0.004399 | PASS |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\YieldSeekerOptExperiment\results\yieldseeker_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\YieldSeekerOptExperiment\results\yieldseeker_family_proxy_details.csv`
