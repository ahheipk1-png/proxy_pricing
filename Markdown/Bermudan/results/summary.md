# Bermudan Option Proxy Experiment

Monthly-exercise Bermudan put proxy using Sobol MC dynamic programming
and SciPy PCHIP continuation curves.

| Method | Worst max relative error | Average p99 | Average MAE |
|---|---:|---:|---:|
| pchip_dynamic_programming | 8.300% | 1.924% | 0.001315 |

Benchmark: independent Cox-Ross-Rubinstein Bermudan tree with exercise
allowed every 24 tree steps.

## Test Cases

| Case | Strike | Vol | Rate | Dividend yield |
|---|---:|---:|---:|---:|
| base_put | 100.00 | 0.240 | 0.045 | 0.015 |
| low_vol | 100.00 | 0.160 | 0.045 | 0.015 |
| high_vol | 100.00 | 0.360 | 0.045 | 0.015 |
| deep_itm | 115.00 | 0.240 | 0.045 | 0.015 |
| dividend_rich | 100.00 | 0.280 | 0.045 | 0.060 |

## Rate And Volatility Term Structures

Bermudan exercise values depend on the timing of rates and volatility, not only
their final integrals. In the single-feature term-structure sensitivity study,
front-loaded versus back-loaded curves with the same total rate and total
variance changed the Bermudan put value by about 5.9%.

For a fixed deterministic curve, keep the proxy one-dimensional in spot at each
exercise date, but use step-specific discount factors, drifts, and variances in
the backward recursion.

If the proxy must generalize across different curves, add remaining curve
summaries such as average rate, effective volatility, near-step variance, and
front/back volatility slope before trying raw curve knots.
