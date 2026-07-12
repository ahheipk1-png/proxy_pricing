# European Vol/Rate Feature Proxy

This experiment extends the European proxy from one feature to three features:

- spot represented by the Black-Scholes `d1` coordinate
- volatility
- risk-free rate

The goal is not to use Black-Scholes as the production method. Black-Scholes is
used here as a clean benchmark so we can see whether the proxy surface learns
the correct dependence on volatility and rate.

## Setup

- strike: `100`
- dividend yield: `2.00%`
- maturity: `1` year
- training domain: `d1 in [-3.5, 3.5]`, `vol in [8.00%, 50.00%]`, `rate in [-1.00%, 10.00%]`
- training grid: `19 x 7 x 7` = `931` Chebyshev-spaced states
- test grid: `61 x 13 x 13` = `10309` uniform states
- shifted Sobol MC paths per state: `4,096`
- relative error denominator: `max(true value, 0.01)`
- elapsed seconds: `0.4`

## MC Label Quality

| Option | Max % Label Error | P99 % Label Error | Avg % Label Error | Label MAE | Label Max Abs |
|---|---:|---:|---:|---:|---:|
| call | 0.017% | 0.015% | 0.002% | 0.003542 | 0.035830 |
| put | 0.002% | 0.002% | 0.000% | 0.000066 | 0.000657 |

## Best Results By Group

| Option | Label Source | Best Method | Coordinate | Terms | Max % Error | P99 % Error | Avg % Error | MAE | Fit Seconds |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| call | exact labels | log sparse Chebyshev degree 7 | d1_vol_rate | 120 | 0.489% | 0.485% | 0.191% | 0.077247 | 0.0012 |
| call | shifted Sobol MC labels | log sparse Chebyshev degree 7 | d1_vol_rate | 120 | 0.493% | 0.487% | 0.191% | 0.076906 | 0.0011 |
| put | exact labels | log sparse Chebyshev degree 7 | d1_vol_rate | 120 | 0.490% | 0.483% | 0.204% | 0.044435 | 0.0011 |
| put | shifted Sobol MC labels | log sparse Chebyshev degree 7 | d1_vol_rate | 120 | 0.489% | 0.482% | 0.204% | 0.044339 | 0.0013 |

## Full Method Comparison

| Option | Label Source | Method | Coordinate | Terms | Max % Error | P99 % Error | Avg % Error | MAE |
|---|---|---|---|---:|---:|---:|---:|---:|
| call | exact labels | log sparse Chebyshev degree 7 | d1_vol_rate | 120 | 0.489% | 0.485% | 0.191% | 0.077247 |
| call | exact labels | log sparse Chebyshev degree 5 | d1_vol_rate | 56 | 0.521% | 0.506% | 0.190% | 0.069131 |
| call | exact labels | log anisotropic tensor Chebyshev 9x3x3 | d1_vol_rate | 160 | 2.407% | 2.407% | 1.165% | 0.412874 |
| call | exact labels | log sparse Chebyshev degree 9 | d1_vol_rate | 220 | 4.674% | 4.670% | 1.502% | 0.539615 |
| call | exact labels | log sparse Chebyshev degree 7 | spot_vol_rate | 120 | 658.315% | 110.803% | 21.793% | 12.818485 |
| call | exact labels | log sparse Chebyshev degree 7 | logm_vol_rate | 120 | 187.347% | 152.345% | 39.086% | 16.574739 |
| call | exact labels | direct sparse Chebyshev degree 7 | d1_vol_rate | 120 | 1916.627% | 796.960% | 57.235% | 0.089367 |
| call | shifted Sobol MC labels | log sparse Chebyshev degree 7 | d1_vol_rate | 120 | 0.493% | 0.487% | 0.191% | 0.076906 |
| call | shifted Sobol MC labels | log sparse Chebyshev degree 5 | d1_vol_rate | 56 | 0.518% | 0.506% | 0.190% | 0.068982 |
| call | shifted Sobol MC labels | log anisotropic tensor Chebyshev 9x3x3 | d1_vol_rate | 160 | 2.409% | 2.408% | 1.165% | 0.413184 |
| call | shifted Sobol MC labels | log sparse Chebyshev degree 9 | d1_vol_rate | 220 | 4.673% | 4.669% | 1.502% | 0.539547 |
| call | shifted Sobol MC labels | log sparse Chebyshev degree 7 | spot_vol_rate | 120 | 658.268% | 110.794% | 21.792% | 12.816997 |
| call | shifted Sobol MC labels | log sparse Chebyshev degree 7 | logm_vol_rate | 120 | 187.345% | 152.341% | 39.086% | 16.573892 |
| call | shifted Sobol MC labels | direct sparse Chebyshev degree 7 | d1_vol_rate | 120 | 1914.780% | 797.262% | 57.224% | 0.089746 |
| put | exact labels | log sparse Chebyshev degree 7 | d1_vol_rate | 120 | 0.490% | 0.483% | 0.204% | 0.044435 |
| put | exact labels | log sparse Chebyshev degree 5 | d1_vol_rate | 56 | 0.533% | 0.489% | 0.189% | 0.037816 |
| put | exact labels | log anisotropic tensor Chebyshev 9x3x3 | d1_vol_rate | 160 | 2.408% | 2.407% | 1.234% | 0.237859 |
| put | exact labels | log sparse Chebyshev degree 9 | d1_vol_rate | 220 | 4.687% | 4.682% | 1.603% | 0.319762 |
| put | exact labels | log sparse Chebyshev degree 7 | logm_vol_rate | 120 | 158.155% | 122.661% | 33.929% | 7.747209 |
| put | exact labels | direct sparse Chebyshev degree 7 | d1_vol_rate | 120 | 1837.985% | 468.232% | 31.486% | 0.086977 |
| put | exact labels | log sparse Chebyshev degree 7 | spot_vol_rate | 120 | 3405.894% | 3324.249% | 609.913% | 131.367929 |
| put | shifted Sobol MC labels | log sparse Chebyshev degree 7 | d1_vol_rate | 120 | 0.489% | 0.482% | 0.204% | 0.044339 |
| put | shifted Sobol MC labels | log sparse Chebyshev degree 5 | d1_vol_rate | 56 | 0.533% | 0.489% | 0.188% | 0.037801 |
| put | shifted Sobol MC labels | log anisotropic tensor Chebyshev 9x3x3 | d1_vol_rate | 160 | 2.409% | 2.407% | 1.234% | 0.237860 |
| put | shifted Sobol MC labels | log sparse Chebyshev degree 9 | d1_vol_rate | 220 | 4.685% | 4.682% | 1.603% | 0.319832 |
| put | shifted Sobol MC labels | log sparse Chebyshev degree 7 | logm_vol_rate | 120 | 158.150% | 122.655% | 33.928% | 7.746960 |
| put | shifted Sobol MC labels | direct sparse Chebyshev degree 7 | d1_vol_rate | 120 | 1837.717% | 468.098% | 31.481% | 0.086968 |
| put | shifted Sobol MC labels | log sparse Chebyshev degree 7 | spot_vol_rate | 120 | 3405.853% | 3324.203% | 609.907% | 131.366494 |

## Interpretation

The most important feature-engineering result is that `d1` remains the right
spot coordinate even when volatility and rate are state variables. It removes
most of the strike/vol/time scaling from the spot axis, so the remaining
surface in volatility and rate is smoother.

The exact-label runs measure pure approximation error. The shifted-MC-label
runs measure the realistic proxy workflow, where each training value is a
Monte Carlo conditional expectation estimate. If MC errors dominate, the
fix is usually more paths, better importance sampling, or fitting a residual
around a simple analytic/control baseline.

For this 3D European case, the generic candidate to carry forward is the
log sparse Chebyshev proxy on `(d1, vol, rate)`. It is convex to train,
fast to evaluate, and extends naturally to higher-dimensional feature sets
without requiring a full tensor grid.

Diagnostic plot: `C:\codex_proj\proxy_pricing\tmp\european_vol_rate_proxy\european_vol_rate_proxy_slices.png`
Raw metrics CSV: `C:\codex_proj\proxy_pricing\tmp\european_vol_rate_proxy\european_vol_rate_proxy_metrics.csv`
