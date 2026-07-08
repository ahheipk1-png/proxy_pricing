# European Option Grid Test

This sweep tested the no-asymptotic tail-biased MC proxy method across
calls and puts, multiple strikes, volatilities, and maturities.

Training still used Monte Carlo labels only. Black-Scholes was used only
as the diagnostic benchmark.

## Setup

- option types: `['call', 'put']`
- strikes: `[80.0, 100.0, 120.0]`
- vols: `[0.1, 0.2, 0.4]`
- maturities: `[0.5, 1.0, 2.0]`
- time fractions: `[0.2, 0.4, 0.6, 0.8]`
- total option parameter cases: `54`
- total time-slice fits per method: `216`
- state points per fit: `121`
- shifted MC paths per state: `25,000`
- relative error denominator: `max(true_value, 0.01)`
- elapsed seconds: `18.2`

## Aggregate Results By Method

Recommended default after this sweep: `log Chebyshev | d1, degree=7`.

| Method | Detail | Worst Max % Error | P95 Max % Error | Avg P99 % Error | Avg MAE | >1% | >3% | >5% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| log Chebyshev | d1, degree=7 | 1.029% | 0.529% | 0.256% | 0.007530 | 1 | 0 | 0 |
| log B-spline | d1, knots=12 | 1.547% | 0.808% | 0.450% | 0.005582 | 5 | 0 | 0 |
| log polynomial | spot, degree=9 | 19.639% | 3.036% | 0.883% | 0.167660 | 39 | 12 | 6 |

## Worst Individual Rows

| Option | K | Vol | T | t | Method | Max % Error | P99 % Error | MAE |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| call | 100 | 0.40 | 2.00 | 0.40 | log polynomial spot, degree=9 | 19.639% | 19.157% | 6.950880 |
| call | 120 | 0.40 | 2.00 | 0.40 | log polynomial spot, degree=9 | 19.549% | 19.397% | 8.293855 |
| call | 80 | 0.40 | 2.00 | 0.40 | log polynomial spot, degree=9 | 19.505% | 18.869% | 5.564933 |
| call | 120 | 0.40 | 2.00 | 0.80 | log polynomial spot, degree=9 | 9.827% | 9.246% | 3.358373 |
| call | 80 | 0.40 | 2.00 | 0.80 | log polynomial spot, degree=9 | 9.758% | 8.763% | 2.228359 |
| call | 100 | 0.40 | 2.00 | 0.80 | log polynomial spot, degree=9 | 9.684% | 9.421% | 2.757151 |
| call | 120 | 0.40 | 1.00 | 0.20 | log polynomial spot, degree=9 | 3.228% | 2.585% | 0.839673 |
| call | 80 | 0.40 | 1.00 | 0.20 | log polynomial spot, degree=9 | 3.212% | 2.540% | 0.549801 |
| call | 120 | 0.40 | 2.00 | 1.20 | log polynomial spot, degree=9 | 3.182% | 2.524% | 0.822055 |
| call | 100 | 0.40 | 2.00 | 1.20 | log polynomial spot, degree=9 | 3.144% | 2.527% | 0.683961 |
| call | 80 | 0.40 | 2.00 | 1.20 | log polynomial spot, degree=9 | 3.122% | 2.421% | 0.530734 |
| call | 100 | 0.40 | 1.00 | 0.20 | log polynomial spot, degree=9 | 3.007% | 2.699% | 0.667670 |
| call | 100 | 0.10 | 0.50 | 0.40 | log polynomial spot, degree=9 | 2.912% | 2.059% | 0.028783 |
| call | 80 | 0.10 | 0.50 | 0.30 | log polynomial spot, degree=9 | 2.315% | 1.603% | 0.027701 |
| call | 80 | 0.10 | 1.00 | 0.80 | log polynomial spot, degree=9 | 2.236% | 1.546% | 0.026514 |

## Notes

- The sweep uses the same shifted-MC state-labeling method as the earlier
  European benchmark.
- No asymptotic proxy, asymptotic anchor, or asymptotic mixing is used.
- The comparison focuses on whether the method remains stable across
  option parameter combinations.
