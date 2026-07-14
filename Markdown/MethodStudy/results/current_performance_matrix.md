# Current Cross-Option Proxy Performance Matrix

This matrix summarizes the best currently documented method for each option
family. It is meant as a quick starting point for future experiments, not as a
replacement for the option-specific markdown files.

Relative errors use the denominator floor stated in each experiment. This is
important for far-OTM or nearly knocked-out cases where the benchmark value is
only a few cents.

| Option type | Best current method | Test coverage | Worst max relative error | Avg p99 | Main note |
|---|---|---:|---:|---:|---|
| European | log Chebyshev in `d1`, degree 7 | 54 option cases, 216 time slices | 1.029% | 0.256% | MC-trained labels; Black-Scholes only used as benchmark. |
| Asian | adjusted-moneyness PCHIP hybrid | 5 fixing dates, 2D state collapsed to 1D feature | 2.454% | 1.141% | Uses the adjusted-strike representation for fixed day. |
| American | PCHIP continuation spline | American put grid | 2.249% | 0.947% | Shape-preserving continuation avoids oscillatory exercise errors. |
| Bermudan | log-continuation PCHIP dynamic programming | 5 parameter cases, 5 exercise dates | 8.300% | 1.924% | Uses Sobol/LR mixture labels and a Bermudan tree benchmark. |
| Barrier | PCHIP/Akima/Bernstein by barrier variant | 3 barrier types, discrete and continuous | 8.133% to 9.096% | not summarized | Brownian-bridge crossing correction for continuous barriers. |
| Cliquet | bounded-logit Chebyshev on expected-total `z` | 5 reset dates | 3.593% | 0.928% | Global floor/cap bounds make logit targets stable. |
| SLV cliquet | adaptive hybrid | 3-state SLV cliquet grid | 5.067% | 2.760% | Local quadratic near short remaining tenor, anisotropic fit earlier. |
| Basket Asian | PCHIP-calibrated PCA log-factor correction | 10 underlyings, mixed-sign correlations | 5.873% | 1.820% | PCA plus residual calibration beats pure sparse Chebyshev. |
| Basket cliquet | cached Sobol/LR safety proxy for hard cases | 3-underlying SLV generalized cliquets | 11.2% safety-proxy spot check | not summarized | Fitted-only proxy is not yet universal for order-statistic coupons. |
| Single generic exotic | max-first direct/log linear or PCHIP | 106 pipeline configurations | 9.410% | 0.430% | Data-driven payoff pipeline across rainbow, Himalayan, yield seeker, lookback, barrier, and binary families. |
| Basket generic exotic | max-first direct/log linear or PCHIP | 120 basket pipeline configurations | 10.933% | 0.587% | Same generic pipeline with weighted-basket and order-statistic rankings. |
| Autocallable | Akima/PCHIP log-value interpolation | 5 autocallable cases, 4 observation dates | 0.123% | 0.051% | Single spot state conditional on no prior autocall. |
| Random payoff | Akima log-value interpolation | 8 random payoffs, 3 market regimes, 4 times | 0.658% | 0.037% | PCHIP also strong; Chebyshev overfits nonsmooth random payoffs. |

## Practical Defaults

- For smooth one-feature pricing functions, start with SciPy PCHIP.
- Try Akima when the target has many local slope changes but remains a
  one-feature interpolation problem.
- Use sparse Chebyshev when the transformed target is globally smooth and the
  feature is well-conditioned.
- Use bounded/logit transforms when payoffs have known floors and caps.
- For high-dimensional basket products, add payoff-aware features, PCA, and
  likelihood-ratio mixture sampling before increasing polynomial degree.
- Keep a Sobol/LR safety proxy available for hard discontinuous or
  order-statistic basket coupons.
