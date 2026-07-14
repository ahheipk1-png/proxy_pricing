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
- train paths per state label: `16,384` low-discrepancy antithetic paths
- benchmark paths per validation state: `16,384` independent low-discrepancy antithetic paths
- validation uses a separate path-ratio stream at shifted scale states
  so the reported error includes out-of-sample Monte Carlo benchmark noise
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `196.1`

## Accuracy Summary

- PASS: `200`
- WATCH: `0`
- REVIEW: `0`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 0.229% | 0.047% | 0.112% | 0.012904 | 100 | 0 | 0 |
| basket | 100 | 0.559% | 0.107% | 0.337% | 0.030147 | 100 | 0 | 0 |

## Subtype Coverage

- observations: `spot`
- performances: `relative`
- rankings: `weighted_basket`
- aggregations: `sum`
- payoffs: `linear`
- proxy methods selected: `linear, log_linear, logit_linear`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| yield_seeker_4_memory_True_-0.05_monthly_steep | basket | logit_linear | 0.559% | 0.526% | 0.109630 | PASS |
| yield_seeker_4_memory_False_-0.05_monthly_base | basket | logit_linear | 0.543% | 0.519% | 0.128062 | PASS |
| yield_seeker_4_memory_True_-0.15_monthly_high | basket | logit_linear | 0.455% | 0.430% | 0.085303 | PASS |
| yield_seeker_4_memory_False_0.05_even_months_steep | basket | logit_linear | 0.416% | 0.401% | 0.108031 | PASS |
| yield_seeker_4_memory_False_0.05_monthly_defensive | basket | logit_linear | 0.375% | 0.373% | 0.089686 | PASS |
| yield_seeker_4_memory_False_-0.05_even_months_high | basket | logit_linear | 0.347% | 0.335% | 0.089294 | PASS |
| yield_seeker_4_memory_True_0.0_monthly_base | basket | logit_linear | 0.298% | 0.284% | 0.064707 | PASS |
| yield_seeker_4_memory_False_0.1_quarterly_steep | basket | logit_linear | 0.244% | 0.241% | 0.069417 | PASS |
| yield_seeker_4_actual_return_False_-0.05_monthly_steep | basket | logit_linear | 0.242% | 0.232% | 0.085185 | PASS |
| yield_seeker_4_memory_True_0.1_monthly_defensive | basket | logit_linear | 0.242% | 0.214% | 0.046076 | PASS |
| yield_seeker_4_memory_False_0.1_even_months_base | basket | logit_linear | 0.234% | 0.227% | 0.062322 | PASS |
| yield_seeker_4_high_low_False_0.0_monthly_steep | basket | logit_linear | 0.232% | 0.230% | 0.087997 | PASS |
| yield_seeker_1_memory_True_-0.05_monthly_steep | single | logit_linear | 0.229% | 0.217% | 0.053512 | PASS |
| yield_seeker_4_memory_False_0.0_quarterly_high | basket | logit_linear | 0.212% | 0.211% | 0.058282 | PASS |
| yield_seeker_4_memory_False_-0.15_late_steep | basket | logit_linear | 0.208% | 0.202% | 0.062415 | PASS |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\YieldSeekerOptExperiment\results\yieldseeker_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\YieldSeekerOptExperiment\results\yieldseeker_family_proxy_details.csv`
