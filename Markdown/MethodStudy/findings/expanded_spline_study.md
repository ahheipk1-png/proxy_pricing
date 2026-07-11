# Expanded one-feature smoother study

## Decision

Keep **PCHIP as the generic one-feature default**.

Natural cubic interpolation was fractionally better on the product-balanced
accuracy average, but the difference was not economically meaningful:

| Method | Balanced average p99 | Average max for values >= 0.05 | Local overshoot rate |
|---|---:|---:|---:|
| Natural cubic interpolation | 1.799% | 1.163% | 2.01% |
| PCHIP | 1.834% | 1.220% | 0.00% |
| Akima | 1.958% | 1.286% | 0.17% |
| MAKIMA | 1.968% | 1.317% | 0.15% |
| Cubic smoothing spline | 2.098% | 1.440% | 2.49% |
| Linear interpolation | 2.683% | 2.058% | 0.00% |

PCHIP beat natural cubic interpolation in 53 of 99 cases on p99 relative
error; natural cubic won 46. Natural cubic's aggregate advantage was only
0.035 percentage points. PCHIP never left the range of its two adjacent
training labels, while natural cubic did so at 2.01% of validation points.

That is the relevant tie-breaker for PFE. A tiny pointwise improvement does not
compensate for artificial local extrema that can perturb an exposure quantile.

## Test design

- 99 price-surface cases and 17 methods.
- European: calls and puts; 10 parameter sets across strike, volatility,
  maturity, rates, and dividends; three remaining maturities each.
- American: four put parameter sets and three exercise dates each.
- Arithmetic Asian: three parameter sets and three fixing dates each after the
  adjusted-moneyness state reduction.
- Barrier: four parameter sets, two dates, down/up/double knock-out, and
  discrete/continuous monitoring.
- 10,000,000 Monte Carlo scenarios per fitted surface.
- 500,000 independent paths per Asian and barrier benchmark state.
- Closed-form Black-Scholes European benchmarks and finite-difference American
  benchmarks.
- Barrier validation nodes were shifted away from the Chebyshev training nodes.
- Product families received equal aggregate weight.
- Raw relative error used a 0.01 denominator floor. A separate maximum over
  values of at least 0.05 prevented one-cent options from deciding the result.

The detailed case-level output is in
`results/expanded/expanded_method_results.csv`.

## Cross-check of the external recommendation

The external answer was directionally sound:

- PCHIP is especially attractive when Greeks are not required.
- Smoothing splines and P-splines can denoise Monte Carlo labels.
- Linear interpolation is a serious dense-grid baseline.
- MAKIMA, piecewise Chebyshev, and rational interpolation deserve tests.
- PFE must ultimately be validated in outer exposure scenarios, not only with
  pointwise price RMSE.

The experiment adds several qualifications:

1. Ten million total paths still leave state-dependent tail noise because the
   budget is divided across grid states. Exact interpolators can work well when
   common random numbers make that noise smooth.
2. Generic GCV-selected regression smoothers performed poorly in steep log-price
   tails and near exercise/barrier features. They can be made competitive with
   product-specific constraints and loss functions, but that weakens their case
   as one fixed universal default.
3. Pole-free rational interpolation does not imply bounded or shape-preserving
   interpolation. Floater-Hormann produced extreme excursions on continuous
   barrier log-values and is rejected as a default.
4. Natural cubic interpolation is a credible accuracy challenger, but its small
   gain was within case-to-case variation and it introduced local overshoot.

## Why PCHIP has a theoretical advantage

PCHIP is a local piecewise cubic Hermite interpolant. For locally monotone data,
its derivative construction uses limited neighboring secants so that the cubic
does not create a new extremum inside the interval. Local support also prevents
one noisy wing label from changing the entire fitted domain. These are useful
properties for positive option values with linear or zero tails, exercise
features, and non-monotone barrier curves.

The guarantee is not that PCHIP denoises Monte Carlo data. It does not. Common
random numbers, antithetics, control variates, importance sampling, and
state-dependent path allocation remain responsible for label quality.

## PFE qualification

This study selects a price-surface default. A production PFE sign-off should
still pass the candidate proxy through the actual outer scenarios and compare
PFE by date at the required quantiles. The validation loss should weight states
using the outer scenario distribution and the portfolio's positive-exposure
region. Pointwise price accuracy is necessary but not sufficient.

## Primary references

- Fritsch and Carlson, *Monotone Piecewise Cubic Interpolation*:
  https://doi.org/10.1137/0717021
- Reinsch, *Smoothing by Spline Functions*:
  https://doi.org/10.1007/BF02162161
- Eilers and Marx, *Flexible Smoothing with B-splines and Penalties*:
  https://doi.org/10.1214/ss/1038425655
- Floater and Hormann, *Barycentric Rational Interpolation with No Poles and
  High Rates of Approximation*:
  https://doi.org/10.1007/s00211-007-0093-y
- Moler and Ionita, MAKIMA construction:
  https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
