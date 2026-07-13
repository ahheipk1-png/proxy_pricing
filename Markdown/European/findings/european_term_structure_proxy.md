# European Term-Structure Vol/Rate Proxy

This standalone experiment is run by `EuroMain_vol_rate.py`. It keeps
`EuroMain.py` unchanged.

The experiment asks whether a European proxy should use the whole rate and
volatility term structure as features, or whether lower-dimensional effective
features are enough.

For deterministic rates and deterministic volatility, the European terminal
log-price is normal with

```text
R = integral r(t) dt,    Q = integral q(t) dt,    V = integral sigma(t)^2 dt.
```

Therefore the option value depends on the curve shapes only through `R` and
`V`. The tested collapsed feature vector is:

```text
(d1, average rate, effective volatility)
```

where `effective volatility = sqrt(V / T)`.

## Setup

- four piecewise-constant rate buckets over one year
- four piecewise-constant volatility buckets over one year
- training states: `17 d1 nodes x 5 rate levels x 3 rate slopes x 3 rate curvatures x 5 vol levels x 3 vol slopes x 3 vol curvatures = 6,885`
- test states: `81 d1 nodes x 7 named rate curves x 7 named vol curves = 3,969` per option type
- MC labels: shifted one-dimensional Sobol terminal-normal draws with likelihood-ratio correction
- benchmark: generalized Black-Scholes with deterministic term structures
- shifted Sobol paths per state: `2,048`
- elapsed seconds: `4.8`

## MC Label Quality

| Option | Max % Label Error | P99 % Label Error | Avg % Label Error | MAE | Max Abs |
|---|---:|---:|---:|---:|---:|
| call | 0.019% | 0.017% | 0.003% | 0.004856 | 0.050781 |
| put | 0.003% | 0.003% | 0.001% | 0.000264 | 0.002573 |

## Best Results

| Option | Label Source | Best Method | Feature Kind | Terms | Max % Error | P99 % Error | Avg % Error | MAE |
|---|---|---|---|---:|---:|---:|---:|---:|
| call | exact labels | collapsed effective curve, degree 5 | collapsed | 56 | 0.525% | 0.496% | 0.169% | 0.051141 |
| call | shifted Sobol MC labels | collapsed effective curve, degree 5 | collapsed | 56 | 0.535% | 0.498% | 0.170% | 0.051539 |
| put | exact labels | collapsed effective curve, degree 5 | collapsed | 56 | 0.569% | 0.499% | 0.160% | 0.027627 |
| put | shifted Sobol MC labels | collapsed effective curve, degree 5 | collapsed | 56 | 0.565% | 0.495% | 0.160% | 0.027552 |

## Full Method Comparison

| Option | Label Source | Method | Feature Kind | Terms | Max % Error | P99 % Error | Avg % Error | MAE |
|---|---|---|---|---:|---:|---:|---:|---:|
| call | exact labels | collapsed effective curve, degree 5 | collapsed | 56 | 0.525% | 0.496% | 0.169% | 0.051141 |
| call | exact labels | collapsed effective curve, degree 7 | collapsed | 120 | 4.846% | 4.648% | 1.282% | 0.473021 |
| call | exact labels | curve stats, degree 3 | curve_stats | 120 | 5.343% | 5.287% | 1.750% | 0.603671 |
| call | exact labels | full curve knots, degree 3 | full_curve | 220 | 12.521% | 8.847% | 2.387% | 0.638855 |
| call | exact labels | curve stats, degree 4 | curve_stats | 330 | 44.174% | 42.428% | 13.424% | 4.975950 |
| call | exact labels | raw spot plus full curve, degree 3 | raw_spot_full_curve | 220 | 502.122% | 476.110% | 115.267% | 77.319157 |
| call | shifted Sobol MC labels | collapsed effective curve, degree 5 | collapsed | 56 | 0.535% | 0.498% | 0.170% | 0.051539 |
| call | shifted Sobol MC labels | collapsed effective curve, degree 7 | collapsed | 120 | 4.834% | 4.749% | 1.284% | 0.490645 |
| call | shifted Sobol MC labels | curve stats, degree 3 | curve_stats | 120 | 5.349% | 5.284% | 1.751% | 0.604871 |
| call | shifted Sobol MC labels | full curve knots, degree 3 | full_curve | 220 | 12.629% | 8.823% | 2.389% | 0.641853 |
| call | shifted Sobol MC labels | curve stats, degree 4 | curve_stats | 330 | 47.364% | 45.366% | 13.941% | 5.167060 |
| call | shifted Sobol MC labels | raw spot plus full curve, degree 3 | raw_spot_full_curve | 220 | 502.176% | 475.969% | 115.270% | 77.322689 |
| put | exact labels | collapsed effective curve, degree 5 | collapsed | 56 | 0.569% | 0.499% | 0.160% | 0.027627 |
| put | exact labels | curve stats, degree 3 | curve_stats | 120 | 4.904% | 4.786% | 1.654% | 0.314198 |
| put | exact labels | collapsed effective curve, degree 7 | collapsed | 120 | 4.874% | 4.807% | 1.380% | 0.286032 |
| put | exact labels | full curve knots, degree 3 | full_curve | 220 | 8.493% | 8.028% | 2.935% | 0.668422 |
| put | exact labels | curve stats, degree 4 | curve_stats | 330 | 88.930% | 86.747% | 27.361% | 5.880579 |
| put | exact labels | raw spot plus full curve, degree 3 | raw_spot_full_curve | 220 | 351.668% | 282.910% | 50.361% | 13.716771 |
| put | shifted Sobol MC labels | collapsed effective curve, degree 5 | collapsed | 56 | 0.565% | 0.495% | 0.160% | 0.027552 |
| put | shifted Sobol MC labels | curve stats, degree 3 | curve_stats | 120 | 4.905% | 4.785% | 1.654% | 0.314263 |
| put | shifted Sobol MC labels | collapsed effective curve, degree 7 | collapsed | 120 | 4.928% | 4.847% | 1.396% | 0.292575 |
| put | shifted Sobol MC labels | full curve knots, degree 3 | full_curve | 220 | 8.531% | 7.926% | 2.917% | 0.664900 |
| put | shifted Sobol MC labels | curve stats, degree 4 | curve_stats | 330 | 92.535% | 90.318% | 28.249% | 6.070926 |
| put | shifted Sobol MC labels | raw spot plus full curve, degree 3 | raw_spot_full_curve | 220 | 351.248% | 282.559% | 50.314% | 13.701064 |

## Worst Named Term-Structure Cases

These rows use the best shifted-MC model for each option type.

| Option | Rate Curve | Vol Curve | Max % Error | P99 % Error | Avg % Error | MAE |
|---|---|---|---:|---:|---:|---:|
| put | downward | flat_mid | 0.565% | 0.562% | 0.285% | 0.050408 |
| put | flat_mid | flat_mid | 0.565% | 0.562% | 0.285% | 0.050408 |
| put | upward | flat_mid | 0.565% | 0.562% | 0.285% | 0.050408 |
| call | downward | flat_mid | 0.535% | 0.534% | 0.251% | 0.062847 |
| call | flat_mid | flat_mid | 0.535% | 0.534% | 0.251% | 0.062847 |
| call | upward | flat_mid | 0.535% | 0.534% | 0.251% | 0.062847 |
| put | flat_high | flat_mid | 0.517% | 0.514% | 0.238% | 0.040195 |
| call | flat_high | flat_mid | 0.517% | 0.515% | 0.236% | 0.055685 |
| put | downward | flat_low | 0.500% | 0.498% | 0.209% | 0.018300 |
| put | flat_mid | flat_low | 0.500% | 0.498% | 0.209% | 0.018300 |
| put | upward | flat_low | 0.500% | 0.498% | 0.209% | 0.018300 |
| call | humped | flat_mid | 0.499% | 0.497% | 0.222% | 0.054026 |

## Shape Invariance Check

The check below compares upward and downward term structures with the same
average rate and the same integrated variance. For a European option, the
values should be identical up to floating-point noise.

| Option | Max Absolute Difference |
|---|---:|
| call | 0.000000000000 |
| put | 0.000000000000 |

## Conclusion

For European options under deterministic rate and volatility curves, do not
feed the whole term structure into the proxy by default. The generic and
more stable feature set is `(d1, average rate, effective volatility)`, where
`d1` is computed from integrated drift and integrated variance.

Raw term-structure knots are not wrong, but they add redundant dimensions.
That makes sparse regression work harder and is a bad habit to carry into
higher-dimensional exotics unless the payoff really observes the path at
intermediate dates.

Diagnostic plot: `C:\codex_proj\proxy_pricing\tmp\euro_vol_rate_term_structure\euro_vol_rate_term_structure_slices.png`
Metrics CSV: `C:\codex_proj\proxy_pricing\tmp\euro_vol_rate_term_structure\euro_vol_rate_term_structure_metrics.csv`
Scenario CSV: `C:\codex_proj\proxy_pricing\tmp\euro_vol_rate_term_structure\euro_vol_rate_term_structure_scenarios.csv`
