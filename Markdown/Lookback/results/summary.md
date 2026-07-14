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
- train paths per state label: `16,384` low-discrepancy antithetic paths
- benchmark paths per validation state: `16,384` independent low-discrepancy antithetic paths
- validation uses a separate path-ratio stream at shifted scale states
  so the reported error includes out-of-sample Monte Carlo benchmark noise
- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation
- selected proxy: lower validation max/p99 error candidate
- elapsed seconds: `348.0`

## Accuracy Summary

- PASS: `82`
- WATCH: `26`
- REVIEW: `92`

| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 100 | 27.550% | 6.125% | 21.575% | 0.011480 | 61 | 12 | 27 |
| basket | 100 | 29.169% | 9.930% | 19.338% | 0.101306 | 21 | 14 | 65 |

## Subtype Coverage

- observations: `lookback`
- performances: `relative`
- rankings: `weighted_basket`
- aggregations: `maximum`
- payoffs: `option`
- proxy methods selected: `linear, log_linear, log_pchip, logit_linear, logit_pchip, nearest, pchip`

## Worst Cases

| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |
|---|---|---|---:|---:|---:|---|
| lookback_4_quarterly_maximum_modified_floating_tail3_normal_1.1_call | basket | log_pchip | 29.169% | 29.168% | 0.016274 | REVIEW |
| lookback_1_full_maximum_floating_strike_tail1_wide_1.0_call | single | linear | 27.550% | 27.406% | 0.006171 | REVIEW |
| lookback_1_after_q1_maximum_floating_strike_tail1_wide_1.0_call | single | linear | 27.550% | 27.406% | 0.006171 | REVIEW |
| lookback_1_back_half_maximum_floating_strike_tail1_wide_1.0_call | single | linear | 27.550% | 27.406% | 0.006171 | REVIEW |
| lookback_1_quarterly_maximum_floating_strike_tail1_wide_1.0_call | single | linear | 27.550% | 27.406% | 0.006171 | REVIEW |
| lookback_4_back_half_minimum_fixed_strike_tail3_uncapped_1.1_call | basket | nearest | 22.447% | 20.730% | 0.106380 | REVIEW |
| lookback_4_back_half_minimum_fixed_strike_tail1_normal_0.9_call | basket | nearest | 22.182% | 22.084% | 0.162371 | REVIEW |
| lookback_1_full_maximum_floating_ratio_tail1_normal_1.0_call | single | linear | 21.839% | 21.575% | 0.004988 | REVIEW |
| lookback_1_after_q1_maximum_floating_ratio_tail1_normal_1.0_call | single | linear | 21.839% | 21.575% | 0.004988 | REVIEW |
| lookback_1_back_half_maximum_floating_ratio_tail1_normal_1.0_call | single | linear | 21.839% | 21.575% | 0.004988 | REVIEW |
| lookback_1_quarterly_maximum_floating_ratio_tail1_normal_1.0_call | single | linear | 21.839% | 21.575% | 0.004988 | REVIEW |
| lookback_4_after_q1_trimmed_average_fixed_strike_tail6_wide_1.0_put | basket | log_pchip | 20.552% | 20.115% | 0.012023 | REVIEW |
| lookback_4_back_half_trimmed_average_fixed_strike_tail1_wide_1.1_put | basket | pchip | 20.017% | 20.008% | 0.022748 | REVIEW |
| lookback_1_full_maximum_modified_floating_tail3_normal_1.1_call | single | linear | 19.565% | 18.921% | 0.007551 | REVIEW |
| lookback_1_after_q1_maximum_modified_floating_tail3_normal_1.1_call | single | linear | 19.565% | 18.921% | 0.007551 | REVIEW |

## Files

- case CSV: `C:\codex_proj\proxy_pricing\LookbackOptExperiment\results\lookback_family_proxy_cases.csv`
- detail CSV: `C:\codex_proj\proxy_pricing\LookbackOptExperiment\results\lookback_family_proxy_details.csv`
