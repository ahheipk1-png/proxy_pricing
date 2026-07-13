# Feature Engineering Guide For Proxy Pricing

Feature engineering is the step that decides whether a proxy problem is easy,
hard, or impossible. A more powerful fitting method cannot fix a missing state
variable. The practical rule is:

```text
first make the state financially sufficient, then make it numerically easy
```

This note is open for discussion. It is meant to guide future experiments, not
freeze one final feature set.

## State Versus Feature

The **state** is the information needed to price future cash flows.

The **feature vector** is the numerical representation passed to the proxy.

Examples:

| Product | Pricing state | Better proxy features |
|---|---|---|
| European | current spot | `d1`, log-moneyness, forward moneyness |
| Asian | spot plus running sum | adjusted moneyness, remaining-average cushion |
| Barrier | spot plus alive flag | log distance to barrier, alive flag |
| American/Bermudan | spot plus exercise date | log-moneyness, continuation cushion, exact intrinsic max |
| Cliquet | accrued coupon plus market state | accrued floor/cap cushion, expected-total z |
| Basket Asian | all spots plus running basket sum | basket level, running-sum cushion, moments, PCA scores |
| Basket cliquet | accrued coupon, spots, variances | accrued cushion, basket level, dispersion, best/worst/spread, PCA |

## Feature Engineering Workflow

1. Write the payoff and observation schedule.
2. Identify the Markov or pricing-sufficient state.
3. Add status flags: alive, called, knocked out, exercise date.
4. Identify exact payoff regions and bounds.
5. Build dimensionless coordinates: moneyness, log-moneyness, `d1`, accrued
   cushion, normalized variance.
6. Add a baseline if the raw value is too curved.
7. Add basket composition features: level, dispersion, PCA, best/worst/spread.
8. Fit a first proxy.
9. Plot signed residuals against omitted candidate variables.
10. Add only features with a financial, mathematical, or stable empirical
    reason.

## Common Feature Types

| Feature type | Examples | Why useful |
|---|---|---|
| Level | `S/K`, basket level, forward moneyness | Main price direction |
| Log level | `log(S/K)`, log spot vector | GBM is additive in log space |
| Greek-inspired | `d1`, delta bucket, vega scale | Allocates resolution by sensitivity |
| Path state | running sum, accrued coupon, realized average | Restores sufficiency for path dependence |
| Status flag | alive, called, knocked out, exercise date | Separates different payoff rules |
| Bound/cushion | distance to floor, cap, barrier, strike | Highlights transition and exact regions |
| Moment baseline | conditional mean, variance, lognormal baseline | Removes dominant smooth shape |
| Cross-sectional | basket level, dispersion, best, worst, spread | Captures basket composition |
| Dimension reduction | PCA scores, common factor, residual factors | Avoids full tensor grids |
| Model state | variance, local-vol bucket, rate state | Captures future dynamics |

## Product-Specific Notes

### European

Raw spot is sufficient under GBM, but `d1` is a better feature:

```text
d1 = [log(S/K) + (r - q + 0.5 sigma^2) tau] / [sigma sqrt(tau)]
```

Why:

- It is tied to Black-Scholes delta.
- It spreads training nodes across option sensitivity regions.
- It works better than raw spot when maturity and volatility change.

### Asian

The important feature is not spot alone. For an arithmetic Asian call:

```text
K_adj = (N K - A - S) / m
feature = K_adj / S
```

Why:

- `A` is the running sum of past fixings.
- `m` is the number of future fixings.
- The remaining future average must beat `K_adj`.
- This collapses the tested two-state Asian problem into a strong one-feature
  representation.

### Barrier

Use:

- alive flag;
- log distance to lower barrier, `log(S/L)`;
- log distance to upper barrier, `log(U/S)`;
- monitoring type: discrete or continuous.

Why:

- Spot alone is not sufficient once a barrier may already have been hit.
- Price changes rapidly near the barrier.
- Continuous monitoring needs Brownian-bridge crossing information.

### American And Bermudan

Use:

- spot or log-moneyness;
- time/exercise index;
- intrinsic value;
- continuation value target;
- exercise-boundary cushion.

Important rule:

```text
value = max(intrinsic, fitted continuation)
```

The proxy should not learn the exercise max from noisy labels if we already
know the exact max structure.

### Cliquet

Use:

- accrued coupon;
- remaining coupon count;
- local floor/cap;
- global floor/cap cushion;
- expected-total feature;
- bounded-logit target.

Why:

- Floor/cap exact regions are critical.
- The proxy should not waste effort learning flat capped/floored tails.

### Basket Asian

Use:

- current basket level;
- running basket sum;
- remaining-average cushion;
- weighted log spots;
- cross-sectional dispersion;
- PCA scores of centered log spots;
- conditional mean and variance of future average;
- moment-matched lognormal baseline;
- residual target.

Why:

- Same basket level can hide different compositions.
- Correlation and dispersion affect future basket distribution.
- The moment baseline removes the largest smooth component.

### Basket Cliquet

Use:

- accrued return;
- remaining reset count;
- floor/cap cushion;
- basket level;
- cross-sectional spread;
- current best/worst indicators;
- PCA scores for log spots and log variances;
- variance state for each underlying;
- local-vol or SLV summaries.

Why:

- Worst-of and best-of coupons have switching surfaces.
- A smooth global fitted proxy may miss ranking changes.
- Hard regions may need cached Sobol/LR safety pricing.

## Feature Transforms And Target Transforms

Features and targets should be designed together.

| Situation | Useful target |
|---|---|
| Positive value, wide range | `log(V + eps)` |
| Known lower and upper bounds | bounded logit |
| Good baseline `B` | `log((V + eps) / (B + eps))` |
| Baseline can be near zero | additive residual |
| Exercise product | continuation value, then exact max |
| Floor/cap product | normalized bounded value |

A good feature can make the target nearly flat. A poor feature forces the
regressor to learn avoidable curvature.

## How To Detect Missing Features

After fitting, plot signed residuals against candidate omitted variables.

Warning signs:

- residual trends with running sum after using spot only;
- residual trends with distance to barrier;
- residual trends with accrued floor/cap cushion;
- basket residuals trend with dispersion;
- basket cliquet residuals trend with best-worst spread;
- SLV residuals trend with variance after using spot only;
- errors cluster around exercise boundary or barrier boundary.

If residuals show stable structure, the issue is probably feature engineering,
not the choice of interpolation method.

## Avoiding Ad Hoc Features

A feature is defensible if it comes from at least one of these sources:

1. **State sufficiency**: the payoff or model dynamics require it.
2. **Asymptotics/exact regions**: it controls a known floor, cap, barrier, or
   tail behavior.
3. **Moment approximation**: it enters conditional mean, variance, or a
   baseline.
4. **Validation residuals**: independent errors show stable structure against
   the variable.

This allows product-specific features without turning the methodology into
case-by-case curve fitting.

## Practical Checklist

Before training:

- Does the state determine future payoff distribution under the model?
- Are running sums, accrued coupons, and realized averages included?
- Are alive/called/knocked-out/exercise statuses included?
- Are features dimensionless?
- Are feature ranges scaled consistently?
- Are exact regions encoded or removed?
- Is there a useful baseline?
- For baskets, are level, dispersion, and ranking effects represented?
- For SLV, are spot and variance both represented?
- For rate/volatility term structures, have curves been compressed into
  product-relevant integrals or event-date summaries before using raw knots?
- Do residual plots show structure against omitted variables?
- Does the feature design survive multiple parameter cases?

For deterministic European term structures, integrated rate and integrated
variance are sufficient. For path-dependent products, use event-date summaries
such as discount factors, forward distances, cumulative variances, local barrier
segment variance, or front/back curve slopes before adding every curve bucket as
a separate feature. See `term_structure_feature_notes.md` for the product-by-
product rule.

## Current View

Feature engineering should be treated as the first research lever:

```text
state sufficiency -> financial coordinates -> variance-reduced labels -> fitting method
```

For one-feature products, a good coordinate often makes PCHIP or Akima enough.
For higher-dimensional products, good features, baselines, PCA, and exact
regions usually matter more than increasing polynomial degree.
