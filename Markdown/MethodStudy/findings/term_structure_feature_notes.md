# Rate And Volatility Term-Structure Features

This note explains how to treat deterministic rate and volatility term
structures in the proxy project, especially for products that are otherwise
single-underlying or single-state proxies.

The key rule is:

```text
do not add raw curve knots unless the payoff actually needs the curve shape
```

Raw curve knots increase dimension quickly. If the product only depends on a
few integrals or event-date summaries, those summaries are better features.

## Numerical Shape-Sensitivity Check

`SingleFeatureTermStructureStudy.py` tests pairs of deterministic curves with
the same total rate integral and the same total variance integral. Only the
timing of rates and volatility changes.

The result is:

These percentages are value sensitivities to deterministic curve timing, not
proxy accuracy errors. The proxy-accuracy numbers live in the option-specific
validation summaries.

| Product | Worst shape sensitivity | Interpretation |
|---|---:|---|
| European call | 0.000% | terminal-only, curve shape collapses |
| Random terminal payoff | 0.001% | terminal-only; residual is MC noise |
| Asian arithmetic call | 51.074% | fixing-date distribution strongly depends on curve timing |
| Bermudan put | 5.922% | exercise value depends on when variance arrives |
| American put | 6.067% | early-exercise boundary depends on rate/vol timing |
| Barrier down-out call | 1.077% | local segment variance affects survival |
| Autocallable note | 2.577% | observation-date probabilities and discounting depend on curve timing |

The practical conclusion is:

```text
terminal payoff: collapse to integrated R and V
path/event payoff: use event-date summaries
barrier payoff: include local segment variance
early exercise: use step-specific discount/drift/variance in the recursion
```

## European Options

For a European option under deterministic rates, dividend yield, and volatility,
the terminal log-price is normal:

```text
log(S_T) = log(S_t) + integral_t^T [r(u) - q(u) - 0.5 sigma(u)^2] du
           + integral_t^T sigma(u) dW_u.
```

The stochastic integral is normal with variance:

```text
V(t,T) = integral_t^T sigma(u)^2 du.
```

So the option value only needs:

```text
R(t,T) = integral_t^T r(u) du
Q(t,T) = integral_t^T q(u) du
V(t,T) = integral_t^T sigma(u)^2 du
```

The recommended feature vector is:

```text
(d1_eff, average_rate, effective_volatility)
```

where:

```text
average_rate = R(t,T) / (T - t)
effective_volatility = sqrt(V(t,T) / (T - t)).
```

The new `EuroMain_vol_rate.py` experiment confirms this. Feeding the raw four
rate buckets and four volatility buckets into the proxy was worse than using
the collapsed effective features.

## Asian Options

For an Asian option, the payoff observes intermediate fixing dates. Therefore a
volatility term structure cannot always be collapsed to final integrated
variance.

For a fixed market curve, the state can still be:

```text
(spot, running sum, fixing index)
```

and the proxy can remain one-dimensional after using the adjusted-moneyness
coordinate. But the coordinate should use time-dependent quantities:

```text
discount factor to payoff date
forward growth to each remaining fixing
covariance matrix of future fixing log-prices
effective variance of the average or geometric-average control variate
```

If the curve itself varies across trades or scenarios, add low-dimensional curve
features rather than raw knots. Good candidates are:

```text
average remaining rate
average remaining forward volatility
front/back volatility slope over remaining fixing dates
variance of the arithmetic-average control variate
```

## American And Bermudan Options

For American and Bermudan options, the exercise rule depends on the value of
waiting. With deterministic term structures, the backward recursion should use
step-specific discount factors, drifts, and volatilities:

```text
df_i = exp(- integral_{t_i}^{t_{i+1}} r(u) du)
drift_i = integral_{t_i}^{t_{i+1}} [r(u) - q(u) - 0.5 sigma(u)^2] du
variance_i = integral_{t_i}^{t_{i+1}} sigma(u)^2 du.
```

If the curve is fixed, the continuation-value proxy at each exercise date can
remain a one-dimensional function of spot. The curve is part of the model
configuration, not a state variable.

If the curve varies across scenarios, use curve summaries at each exercise
date:

```text
remaining average rate
remaining effective volatility
front/back volatility slope
discounted strike or forward-moneyness coordinate
```

Do not start with all curve knots unless those summaries fail residual tests.

## Barrier Options

Barrier options observe the path relative to barriers. Discrete monitoring uses
the simulated values at monitoring dates. Continuous monitoring needs local
segment variances for Brownian-bridge survival probabilities.

With deterministic term structures, the simulator should use:

```text
variance_i = integral_{t_i}^{t_{i+1}} sigma(u)^2 du
drift_i = integral_{t_i}^{t_{i+1}} [r(u) - q(u) - 0.5 sigma(u)^2] du
```

For a fixed curve, a barrier proxy can still be one-dimensional in spot,
conditioned on barrier status and monitoring date.

If the curve varies across scenarios, useful features are:

```text
distance to barrier in log space
cumulative remaining variance
near-barrier local variance over the next monitoring interval
remaining average rate
front/back volatility slope
```

For continuous barriers, local variance near the next segment is more important
than far-end curve shape.

## Autocallable And Other Observation-Date Products

Autocallables and coupon notes depend on event dates. Curve shape matters
through the discount factors, forwards, and variances to each observation date.

Use event-date summaries:

```text
discount factors to coupon/call dates
forward distances to call and coupon barriers
cumulative variances to observation dates
front/back volatility slope between observation dates
```

If there are many dates, compress these into a few interpretable factors:

```text
near-date average
middle-date average
far-date average
slope
curvature
```

## Cliquet And Reset-Return Products

Cliquet payoffs observe returns over reset periods. A deterministic volatility
term structure matters through each reset period variance, not just final
integrated variance.

For fixed curves, the reset-date proxy can stay low-dimensional if the reset
period parameters are part of the model configuration. Across changing curves,
use summaries aligned to reset periods:

```text
discount factor to maturity
forward growth per reset period
variance per reset period
remaining average reset variance
front/back reset variance slope
accrued floor/cap cushion
```

Do not use only terminal effective volatility for cliquets; it loses the timing
information that determines which local caps and floors bind.

## Practical Decision Rule

Use this hierarchy:

```text
1. If the payoff only sees terminal S_T, use integrated R and V.
2. If the payoff sees scheduled dates, use values of R and V to those dates.
3. If the payoff is barrier-sensitive, include local variance near the barrier interval.
4. If residual plots still show structure by curve shape, then add slope/curvature factors.
5. Use raw curve knots only after lower-dimensional summaries fail.
```

This keeps single-feature products single-feature when the market curve is
fixed, and it prevents unnecessary dimensionality when the curve varies.
