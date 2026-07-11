# Barrier option proxy findings

## Scope

The experiment covers:

- down-and-out, up-and-out, and double-knock-out calls;
- monthly discrete monitoring and continuous monitoring;
- zero rebate;
- 10,000,000 training scenarios per fitted date;
- 500,000 independent benchmark paths per validation state.

Zero-rebate knock-ins follow from in/out parity under the same monitoring rule:

```text
V_knock_in = V_vanilla - V_knock_out.
```

If historical monitoring has already triggered the barrier, the alive/hit flag
must be included in the state. The one-feature curves here condition on the
contract still being alive.

## Continuous-monitoring correction

For log spots `x` and `y` over a time step with variance
`q = sigma^2 dt`, the conditional survival probability above a lower log
barrier `b` is

```text
P(min bridge > b | x,y)
  = 1 - exp(-2 (x-b)(y-b) / q),  x>b and y>b.
```

The upper-barrier formula replaces `(x-b)(y-b)` by `(b-x)(b-y)`.
The implementation multiplies these segment survival probabilities and uses
the product as a conditional payoff weight. This integrates out the
within-step crossing indicator and is smoother than drawing an additional
Bernoulli crossing event.

For a double barrier, the survival probability is the absorbing transition
density on the log interval divided by the unconstrained transition density.
The absorbing density is evaluated with a truncated method-of-images series.
This accounts for crossings of either barrier and interactions between the two
boundaries.

## Sampling and fitting

- Spot nodes use a log-scale Chebyshev grid over the alive region.
- Common random numbers are reused across spot states so training noise varies
  smoothly with spot.
- Antithetic normals reduce transition noise.
- Validation uses shifted interior nodes; it is not nested inside the training
  grid.
- The positive price is fitted in log space.
- PCHIP is the generic one-feature default; Akima, Chebyshev ridge, and
  Bernstein ridge are comparison methods.

## Default results

| Variant | PCHIP raw worst | Worst for values >= 0.05 |
|---|---:|---:|
| Down-out discrete | 2.253% | 2.087% |
| Down-out continuous | 2.531% | 1.762% |
| Up-out discrete | 9.096% | 4.853% |
| Up-out continuous | 8.808% | 5.074% |
| Double-out discrete | 8.387% | 4.999% |
| Double-out continuous | 8.148% | 4.916% |

The large raw maxima occur at values around one or two cents. On values of at
least five cents, all but one variant are below 5%; up-out continuous is
5.074%.

The expanded 99-case one-dimensional study also varied barrier width,
volatility, and maturity. PCHIP remained close to natural cubic interpolation
on accuracy while avoiding local overshoot, so it remains the production
default.

## Limitations

- The bridge formulas are exact for constant-parameter GBM segments. Under
  local or stochastic volatility they become approximations unless the
  segment variance treatment is refined.
- Continuous double-barrier survival uses a finite image series and should be
  convergence-checked for extremely narrow barriers or large time steps.
- Nonzero rebates require their own hit-time or terminal rebate treatment.
- A knock-in after monitoring has begun needs the historical hit flag.

## Primary references

- Broadie, Glasserman, and Kou, *A Continuity Correction for Discrete Barrier
  Options*: https://www.columbia.edu/~sk75/mfBGK.pdf
- Giles, *Multilevel Monte Carlo Path Simulation* and Brownian-bridge barrier
  treatment: https://arxiv.org/abs/0904.1157
