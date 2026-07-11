# SLV Cliquet Proxy Findings

## Product and model

The product is the same monthly cliquet used in the GBM experiment:

- 12 monthly simple returns over one year
- local floor/cap: -2% / 4%
- global floor/cap: 0% / 20%
- notional: 100

The underlying follows an illustrative stochastic-local-volatility model:

```text
dS/S = (r-q) dt + L(S) sqrt(v) dW_S
dv   = kappa(theta-v) dt + xi sqrt(v) dW_v
corr(dW_S,dW_v) = rho
```

`L(S)` is a bounded tanh leverage function with higher local volatility at low
spots. The variance process uses full-truncation Euler. This is a research model,
not a leverage surface calibrated to market vanillas.

## State and proxy

At a reset date the Markov state is:

```text
(accrued clipped return, current spot, current variance)
```

The most useful feature is a normalized expected-total return:

```text
z = (accrued + remaining * frozen_coupon_mean - global_midpoint)
    / sqrt(remaining * frozen_coupon_variance)
```

The target is the logit of price normalized to its known discounted global
floor/cap bounds. Exact floor and cap tails are imposed before regression.

The robust default is maturity-adaptive:

- 9-12 resets remaining: local quadratic regression in `(z, log(S/S0), log(v/theta))`
- 3-6 resets remaining: anisotropic Chebyshev regression, degree 19 in `z` and
  degree 3 in the two SLV coordinates
- maturity: exact payoff

This rule is time based and product structural; it does not select a method
state by state.

## Sampling and benchmark

- Training: about 10 million scenarios per fitted reset date
- Training states: 321 Halton-style states with 75% boundary-focused accrued
  return sampling
- Benchmark: 500,000 paths per validation state
- Variance reduction: antithetic paths and lower-tail likelihood-ratio
  importance sampling
- SLV discretization: two Euler steps per monthly return

Importance sampling is essential late in the deal. It shifts the independent
spot Brownian shocks upward when the expected total is below the global floor,
then applies the exact likelihood ratio. It reduces noise in values only a few
cents above zero without changing the MC expectation.

## Results

The common-random-number rerun's default result is:

| Reset month | Component | Max relative error | p99 relative error | MAE |
|---:|---|---:|---:|---:|
| 0 | local quadratic | 0.778% | 0.536% | 0.00870 |
| 3 | local quadratic | 2.492% | 2.205% | 0.03327 |
| 6 | anisotropic Chebyshev | 4.043% | 3.346% | 0.01475 |
| 9 | anisotropic Chebyshev | 5.067% | 4.952% | 0.01854 |
| 12 | exact | 0.000% | 0.000% | 0.00000 |

An additional two-design study rebuilt 10M-path labels and used 500,000 paths
per benchmark state on each design. The fixed maturity-adaptive hybrid remained
best at 4.795% worst max error. Degrees 13-23, sparse Hermite, local regression,
and Nystrom Matern regression did not improve robustness.

Common random numbers are now reused across training states. They make MC label
noise smooth in state and improve interpolation stability without changing any
individual conditional expectation.

## Conclusion

A one-dimensional accrued-return proxy is insufficient under SLV. Spot and
variance matter because they change both future return skew and the likelihood
of hitting local caps and floors.

The useful generic methodology is:

1. Keep the full Markov state.
2. compress payoff geometry into an expected-total standardized feature;
3. enforce known price bounds through the target transform;
4. sample states around both global payoff transitions;
5. importance-sample rare lower-tail payoffs;
6. use local smoothing when the long-horizon surface is broad, then switch to a
   sparse anisotropic spectral fit as the payoff transitions sharpen.

The reported errors measure proxy error against MC under the same two-step
monthly SLV discretization. They do not measure Euler bias or model calibration
error. A production study should repeat the benchmark with finer time steps and
a calibrated leverage surface.
