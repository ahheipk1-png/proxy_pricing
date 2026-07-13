# Random Option Proxy Experiment

Fixed-seed random piecewise-linear terminal payoffs under GBM. This is a
generic one-feature stress test for the interpolation proxy rather than a
named exchange-traded product.

Best method on this grid: `akima`.

| Method | Worst max relative error | Average p99 | Average MAE |
|---|---:|---:|---:|
| pchip | 0.741% | 0.067% | 0.000244 |
| akima | 0.658% | 0.037% | 0.000171 |
| chebyshev | 55.459% | 4.911% | 0.246215 |

## Test Grid

- Payoff cases: 8 random piecewise-linear terminal payoffs.
- Market cases: 3 volatility regimes.
- Time fractions: 0.0, 0.25, 0.5, 0.75.

## Rate And Volatility Term Structures

These random options are terminal-payoff products. Like European options, they
depend on deterministic rate and volatility curves only through integrated rate
and integrated variance.

The single-feature term-structure sensitivity study confirmed this: changing
front-loaded versus back-loaded curve shape with the same total rate and total
variance produced only about 0.001% relative difference, consistent with Monte
Carlo noise.

For terminal random payoffs, use:

```text
d1/effective-moneyness, average rate, effective volatility
```

Do not add raw rate or volatility curve knots unless the payoff itself becomes
path-dependent or observation-date dependent.
