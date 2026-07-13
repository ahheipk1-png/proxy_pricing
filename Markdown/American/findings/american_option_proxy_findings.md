# American Put Proxy Findings

## Product and state

The test product is a one-year American put under GBM:

```text
K = 100, r = 5%, q = 2%, sigma = 20%
```

A put is used because a non-dividend American call has no early-exercise
premium. The Markov state is only `(spot, time)`.

## Training method

The proxy is trained by Monte Carlo dynamic programming on 100 exercise dates.
At each backward step:

1. use 121 log-spaced Chebyshev state nodes between spots 40 and 220;
2. simulate antithetic one-step GBM transitions;
3. evaluate the next-date proxy;
4. estimate discounted continuation values;
5. fit the continuation surface;
6. impose exercise exactly as `max(K-S, continuation)`.

The complete backward pass uses 10,006,700 one-step MC transitions.

## Benchmark

The independent benchmark is a projected implicit finite-difference American
put solver with:

- 4,000 time steps
- 2,000 spot steps on `[0, 400]`
- projected SOR enforcement of the early-exercise constraint

At spot 100 and time zero, the finite-difference value is about 6.6597. A
separate 4,000-step CRR check gives about 6.6605.

## Method comparison

Global polynomial errors compound during backward induction. A small
continuation error at one date becomes an input at every earlier date.
Exponentiating a fitted log value is especially unstable.

| Method | Worst max relative error | Average p99 | Average MAE |
|---|---:|---:|---:|
| Direct Chebyshev degree 9 | 8695.717% | 6194.163% | 0.360034 |
| Linear spline | 126.023% | 110.024% | 0.096668 |
| Shape-preserving PCHIP | **2.249%** | **0.947%** | **0.001479** |
| Akima spline | 3.225% | 1.675% | 0.002033 |
| Log Chebyshev degree 7 | 5541.197% | 4423.119% | 148.866438 |

The very large percentage errors for failed methods occur mainly in small-value
OTM regions, but their absolute and near-boundary errors are also too large to
accept.

The independent-seed standalone PCHIP run has:

| Time | Max relative error | p99 relative error | MAE |
|---:|---:|---:|---:|
| 0.0 | 3.515% | 3.387% | 0.00653 |
| 0.2 | 4.419% | 4.231% | 0.00705 |
| 0.4 | 5.289% | 5.077% | 0.00686 |
| 0.6 | 1.935% | 1.753% | 0.00264 |
| 0.8 | 0.559% | 0.512% | 0.00139 |
| 1.0 | 0.000% | 0.000% | 0.00000 |

## Conclusion

The European log-Chebyshev default should not be carried over mechanically.
European values are globally smooth, while the American value has a moving
exercise kink and recursively fitted continuation values.

The generic American methodology is:

1. regress continuation, not the already kinked option value;
2. impose intrinsic value exactly;
3. use a shape-preserving local interpolant;
4. sample both wings densely;
5. validate the full backward recursion, not just one regression step.

Akima is viable, but PCHIP is more accurate and stable around the exercise
boundary because its monotonicity-preserving slopes suppress local overshoot.
The remaining discrepancy includes the difference between 100-date MC exercise
and the near-continuous finite-difference benchmark.

## Rate And Volatility Term Structures

American values depend on when rates and volatility arrive because the exercise
boundary is a time-dependent free boundary. In the term-structure sensitivity
study, front-loaded versus back-loaded curves with the same total rate and
total variance changed the American put value by about 6%.

For a fixed deterministic curve, the proxy at each exercise time can still be a
one-dimensional function of spot. The backward recursion should use
step-specific quantities:

```text
df_i = exp(- integral r(u) du over the step)
drift_i = integral [r(u)-q(u)-0.5 sigma(u)^2] du over the step
variance_i = integral sigma(u)^2 du over the step
```

If the proxy must generalize across curves, add curve summaries rather than
raw curve knots:

```text
remaining average rate
remaining effective volatility
near-step variance
front/back volatility slope
forward-moneyness or discounted-strike coordinate
```
