# Single-Feature Term-Structure Sensitivity Study

This study checks whether deterministic rate and volatility curve shape can
be collapsed for the other single-underlying proxy products.

Each scenario pair has the same total rate integral and the same total
variance integral. Only the timing of rates and volatility changes.

Important: these percentages are value sensitivities to changing the
curve shape. They are not proxy errors. Proxy accuracy is reported in
the option-specific validation summaries.

## Curves

- flat rate: `[0.04, 0.04, 0.04, 0.04]`
- front-loaded rate: `[0.075, 0.055, 0.025, 0.005]`
- back-loaded rate: `[0.005, 0.025, 0.055, 0.075]`
- flat effective volatility: `[0.265754, 0.265754, 0.265754, 0.265754]`
- front-loaded volatility: `[0.4, 0.3, 0.15, 0.1]`
- back-loaded volatility: `[0.1, 0.15, 0.3, 0.4]`
- total rate integral in each pair: `0.040000`
- total variance integral in each vol-shape pair: `0.070625`
- MC paths for path products: `262,144` antithetic paths
- elapsed seconds: `0.6`

## Worst Shape Sensitivity By Product

| Product | Type | Worst Pair | Front Value | Back Value | Signed Diff | Relative Diff | Feature Recommendation |
|---|---|---|---:|---:|---:|---:|---|
| European call | terminal | rate shape only | 11.546314 | 11.546314 | 0.000000 | 0.000% | terminal integrated rate and variance |
| Random terminal payoff | terminal | vol shape only | 26.416388 | 26.416019 | 0.000369 | 0.001% | terminal integrated rate and variance |
| Asian arithmetic call | path_average | rate+vol shape | 10.154989 | 4.968434 | 5.186555 | 51.074% | event-date forwards, average variance, front/back vol slope |
| Bermudan put | early_exercise | vol shape only | 9.712674 | 9.137450 | 0.575224 | 5.922% | time-step discount/drift/variance; remaining curve summaries if curves vary |
| American put | early_exercise | rate shape only | 9.145226 | 9.735945 | -0.590719 | 6.067% | time-step discount/drift/variance; remaining curve summaries if curves vary |
| Barrier down-out call | barrier | vol shape only | 11.365025 | 11.488752 | -0.123727 | 1.077% | log barrier distance, local next-segment variance, remaining variance |
| Autocallable note | observation_dates | rate+vol shape | 99.128182 | 101.750187 | -2.622005 | 2.577% | discount factors and cumulative variances to observation dates |

## Full Sensitivity Table

| Product | Scenario Pair | Front Value | Back Value | Signed Diff | Relative Diff |
|---|---|---:|---:|---:|---:|
| European call | rate shape only | 11.546314 | 11.546314 | 0.000000 | 0.000% |
| European call | vol shape only | 11.546314 | 11.546314 | 0.000000 | 0.000% |
| European call | rate+vol shape | 11.546314 | 11.546314 | 0.000000 | 0.000% |
| Random terminal payoff | rate shape only | 26.427272 | 26.427272 | 0.000000 | 0.000% |
| Random terminal payoff | vol shape only | 26.416388 | 26.416019 | 0.000369 | 0.001% |
| Random terminal payoff | rate+vol shape | 26.416388 | 26.416019 | 0.000369 | 0.001% |
| Asian arithmetic call | rate shape only | 8.225020 | 7.408979 | 0.816041 | 9.921% |
| Asian arithmetic call | vol shape only | 9.732966 | 5.356530 | 4.376436 | 44.965% |
| Asian arithmetic call | rate+vol shape | 10.154989 | 4.968434 | 5.186555 | 51.074% |
| Bermudan put | rate shape only | 9.062730 | 9.570477 | -0.507746 | 5.305% |
| Bermudan put | vol shape only | 9.712674 | 9.137450 | 0.575224 | 5.922% |
| Bermudan put | rate+vol shape | 9.261454 | 9.208684 | 0.052770 | 0.570% |
| American put | rate shape only | 9.145226 | 9.735945 | -0.590719 | 6.067% |
| American put | vol shape only | 9.778883 | 9.199734 | 0.579149 | 5.922% |
| American put | rate+vol shape | 9.347143 | 9.356652 | -0.009509 | 0.102% |
| Barrier down-out call | rate shape only | 11.463584 | 11.402264 | 0.061320 | 0.535% |
| Barrier down-out call | vol shape only | 11.365025 | 11.488752 | -0.123727 | 1.077% |
| Barrier down-out call | rate+vol shape | 11.384642 | 11.462040 | -0.077398 | 0.675% |
| Autocallable note | rate shape only | 99.372685 | 100.747922 | -1.375237 | 1.365% |
| Autocallable note | vol shape only | 99.781121 | 101.025468 | -1.244347 | 1.232% |
| Autocallable note | rate+vol shape | 99.128182 | 101.750187 | -2.622005 | 2.577% |

## Interpretation

Terminal-only payoffs are invariant to curve shape once total rate and
total variance are fixed. That includes European options and random
terminal payoffs.

Path-dependent products are not invariant. Asian, barrier, early-exercise,
and autocallable values changed when volatility or rates were moved earlier
or later in time. For these products, keeping a one-dimensional spot proxy
is reasonable only when the curve is fixed as part of the model
configuration. If the curve varies across trades or scenarios, add
event-date summaries rather than raw curve knots. This is a feature
engineering conclusion, not a measured proxy-accuracy result.

Practical rule:

```text
terminal payoff: integrated R and V are enough
scheduled payoff: discount factors and variances to event dates
barrier payoff: local segment variance near the barrier matters
early exercise: step-specific discount/drift/variance in the backward recursion
```

CSV: `C:\codex_proj\proxy_pricing\tmp\single_feature_term_structure\single_feature_term_structure_sensitivity.csv`
Plot: `C:\codex_proj\proxy_pricing\tmp\single_feature_term_structure\single_feature_term_structure_sensitivity.png`
