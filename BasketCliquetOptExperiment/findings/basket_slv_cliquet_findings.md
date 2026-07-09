# Three-Underlying SLV Basket Cliquet Findings

## Contract variants

All variants use three SLV underlyings, 12 monthly resets, local floor/cap
`-2% / 4%`, global floor/cap `0% / 20%`, and notional 100.

The three monthly coupon definitions are:

1. `basket_return`: clip the equal-weight basket return;
2. `average_clipped`: average the three individually clipped returns;
3. `worst_of`: clip the worst-performing underlying return.

## Model and state

Each asset has its own stochastic variance and bounded local-volatility leverage.
The market Brownian drivers are correlated, while each asset has a negative
correlation between its own spot and variance shocks.

At reset dates the complete state is seven-dimensional:

```text
(accrued return, S1, S2, S3, v1, v2, v3)
```

Unlike the single-name GBM cliquet, spots and variances cannot be discarded.
They determine future leverage, stochastic variance, basket dependence, and
local floor/cap probabilities.

## Training and benchmark

- 481 low-discrepancy, boundary-enriched training states
- approximately 10 million paths across each fitted reset-date state set
- 31 independent validation states
- 500,000 benchmark paths per validation state
- two full-truncation Euler steps per monthly coupon period
- antithetic sampling plus a defensive mixture importance sampler

The feature set includes standardized lower and upper global-bound cushions,
plus log spot and log variance coordinates. Conditional one-period coupon
moments are used to align states across maturities.

## Methods tried

The experiment compared:

- local quadratic regression on summary features;
- local linear and quadratic regression on the full state;
- sparse anisotropic Chebyshev regression;
- Gaussian radial-basis kernel regression;
- a two-layer tanh neural-network ensemble;
- a maturity-adaptive local/spectral blend.

No single high-dimensional method was uniformly best. Local regression was
strong early, while the sparse spectral fit was better close to maturity.
RBF and the small neural ensemble were not competitive with this amount of
state coverage.

## Development ensemble

The development proxy blended a local and a sparse spectral estimate:

```text
V_proxy = w(m) V_local + (1-w(m)) V_spectral
w(m) = 0.16 + 0.02 m
```

where `m` is the number of remaining monthly coupons. The symmetric
`basket_return` and `average_clipped` products use a summary-feature local
quadratic. The `worst_of` product uses a full-state local quadratic because
asset identity and dispersion matter directly.

On the first development validation design this blend appeared to keep all three
variants inside 5-8%. Two untouched state designs rejected that conclusion. The
blend weights were too dependent on the first validation geometry and are not
the final recommendation.

## Robust validation results

The strongest fixed generic baseline is sparse anisotropic Chebyshev regression
on 2,001 low-discrepancy states. Its max errors on three different validation
state designs were:

| Variant | Development | Validation 2 | Validation 3 | Worst across designs |
|---|---:|---:|---:|---:|
| Basket return | 10.258% | 7.108% | 18.500% | 18.500% |
| Average clipped | 5.624% | 11.566% | 5.745% | 11.566% |
| Worst of | 9.700% | 9.862% | 6.774% | 9.862% |

Increasing coverage to 5,001 states under the same 10M-path budget changed
which coupon variant was easiest but did not uniformly improve worst-case
error. Bagging the 2,001- and 5,001-state proxies also failed to control a third
untouched basket-return tail state.

The current 10M-path budget does not guarantee 5-8% max error across arbitrary
seven-dimensional SLV basket states.

## Literature-inspired residual study

The saved 2,001- and 5,001-state labels were also used to test:

- a moment-matched clipped-normal low-fidelity baseline;
- logit residual Hermite regression;
- local residual smoothing;
- Nystrom Matern kernel regression;
- direct Hermite and direct kernel fits with coupon skewness and floor/cap
  masses;
- inverse-variance weighted sparse Chebyshev fits;
- fixed residual and direct ensembles.

None beat the original unweighted 2,001-state sparse Chebyshev model on maximum
relative error across the development and independent designs. The
moment-matched baseline did not simplify clipped and worst-of SLV coupons
enough; additive residuals leaked small dollar errors into near-zero states,
while logit residuals remained biased. Weighted dense fits sometimes reduced
MAE but worsened the maximum tail error.

This negative result is important: adding a sophisticated smoother cannot
replace state coverage and sufficiently accurate labels in seven dimensions.

## Higher-dimensional conclusion

The generic lesson is not that one fixed regressor wins in every dimension.
The stable workflow is:

1. retain the true Markov state;
2. construct dimensionless payoff-aware features;
3. enforce exact price bounds through a bounded target;
4. cover the state space with low-discrepancy and transition-focused samples;
5. compare global and local smoothers;
6. test model selection on more than one state-space design;
7. keep an untouched high-path benchmark;
8. enrich states adaptively where independent residuals remain large.

The next credible improvement is adaptive state enrichment: add training states
around failed validation neighborhoods, regenerate labels there, and stop only
when a new untouched design passes. A larger path-level neural training program
is another candidate, but the small MLP and random-feature networks tested here
were not competitive.

The results measure proxy error under the same Euler discretization. They do not
measure leverage-surface calibration error or time-discretization bias.
