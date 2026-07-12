# Method Motivation And Selection Guide

This note explains why each proxy-fitting method exists, when it is better than
the alternatives, and what to try next. It is meant to be read before adding a
new method or rerunning a large experiment.

The short version:

| Situation | First method to try | Why |
|---|---|---|
| One feature, clean labels, kinks or barriers possible | PCHIP | Local, shape-preserving, no artificial extrema |
| One feature, clean labels, many slope changes | Akima or MAKIMA | Local cubic slopes react smoothly to uneven local behavior |
| One feature, very smooth global surface | Chebyshev ridge or Bernstein ridge | Regularized global basis denoises MC labels |
| One feature, noisy labels | Smoothing spline or P-spline | Trades exact interpolation for lower variance |
| Early-exercise continuation | PCHIP on continuation, then exact max | Prevents oscillatory exercise boundary errors |
| Known floor/cap payoff | Bounded target plus PCHIP or Chebyshev | Enforces financial bounds by construction |
| Multi-feature smooth proxy | Sparse Chebyshev ridge | Fast, stable, interpretable, convex fit |
| Multi-feature correlated basket | Baseline plus PCA sparse correction | Captures level and composition without full tensor grid |
| Order-statistic basket cliquet | Cached Sobol/LR safety proxy | Switching surfaces are too hard for one smooth fitted surface |

## How To Think About "Better"

"Better" depends on the failure mode.

| Criterion | Meaning |
|---|---|
| Worst relative error | Controls the bad validation point, important for production risk sign-off |
| p99 relative error | Ignores a few pathological points but still focuses on tail quality |
| Signed error smoothness | Reveals missing knots, MC noise, or extrapolation artifacts |
| Shape safety | Avoids negative prices, artificial extrema, or exercise violations |
| Training cost | Time to generate labels and fit the proxy |
| Pricing speed | Time to evaluate the proxy after training |
| Generality | Whether the same idea transfers to the next product |

For PFE and exposure work, shape safety can matter as much as a small average
accuracy improvement. A tiny RMSE gain is not attractive if it creates a fake
local bump that changes an exposure quantile.

## One-Dimensional Methods

### Linear Interpolation

Motivation: linear interpolation is the simplest local baseline. It never
overshoots between two neighboring labels.

Why it can be better:

- Very robust when labels are noisy.
- No artificial curvature.
- Easy to audit.

Why it can be worse:

- Only first-order accurate on smooth curves.
- Produces visible corners even when the true price is smooth.
- Can require a dense grid to reach the same accuracy as cubic methods.

Use it as a sanity check. If a fancy method performs much worse than linear,
the fancy method is probably unstable for that product.

### PCHIP

Motivation: option prices often have monotone regions, flat tails, exercise
kinks, and barrier transitions. PCHIP gives a local cubic interpolant while
limiting slopes to avoid artificial extrema.

Why it can be better:

- Local: one bad label does not distort the whole domain.
- Shape-preserving on monotone data.
- No local overshoot in the expanded one-feature study.
- Good around exercise and barrier features.
- Very fast at pricing time.

Why it can be worse:

- It interpolates MC labels exactly, so it does not denoise by itself.
- If labels are rough because path counts are too low, PCHIP can preserve that
  roughness.
- It is not globally spectral, so very smooth European surfaces may be fit more
  accurately by regularized global methods.

Current role: default one-feature method because it has the best balance of
accuracy, locality, and shape safety.

### Akima And MAKIMA

Motivation: Akima methods build local cubic slopes from nearby secants, with
weights designed to reduce sensitivity to abrupt local slope changes.

Why they can be better:

- Good when the target has locally changing slope but not a strict monotonicity
  requirement.
- Often smoother-looking than PCHIP on mildly irregular data.
- Strong in the Asian adjusted-moneyness experiment.
- Useful for random payoff tests with many shape changes.

Why they can be worse:

- Less strict shape preservation than PCHIP.
- Can introduce small overshoot.
- The slope rule is heuristic rather than directly tied to financial bounds.

Current role: strong challenger for one-feature products, especially when PCHIP
looks too angular or conservative.

### Natural Cubic Interpolation

Motivation: natural cubic interpolation gives a twice-continuously
differentiable curve through all labels, with zero second derivative at the
endpoints.

Why it can be better:

- Very smooth.
- Fractionally best product-balanced p99 in the expanded one-feature study.
- Can give excellent accuracy when the true price surface is smooth and labels
  are clean.

Why it can be worse:

- Global coupling: one label can affect distant parts of the curve.
- Can overshoot between labels.
- Not ideal near barriers, exercise boundaries, or flat payoff tails.

Current role: credible challenger, but not the default because its small
accuracy edge did not compensate for local overshoot risk.

### B-Spline Regression

Motivation: B-splines represent a curve using local polynomial basis functions.
They are better conditioned than raw truncated-power splines.

Why it can be better:

- Can smooth noisy MC labels.
- Local support makes it more stable than high-degree monomials.
- Flexible knot placement can concentrate resolution near kinks.

Why it can be worse:

- Requires choosing knots and regularization.
- If knots miss a barrier or exercise feature, the fit can blur the feature.
- Less automatic than PCHIP for a one-feature production default.

Current role: useful when labels are noisy and exact interpolation is too
literal.

### Smoothing Spline

Motivation: smoothing splines choose a curve that balances label fit and
roughness penalty, often written as

```text
sum_i (y_i - f(x_i))^2 + lambda integral (f''(x))^2 dx.
```

Why it can be better:

- Directly addresses MC label noise.
- Gives a smooth curve without manually choosing every knot.
- Useful when common random numbers are not enough to smooth labels.

Why it can be worse:

- Generic smoothing can blur true kinks.
- GCV-style smoothing parameters may underfit tails or sharp transitions.
- Shape constraints are not automatic.

Current role: a denoising candidate, not a universal default.

### P-Spline

Motivation: P-splines combine a B-spline basis with a penalty on coefficient
differences. They are a practical, stable smoother.

Why it can be better:

- Fast and numerically stable.
- Easy to control smoothness with a penalty.
- More scalable than a full smoothing spline in some settings.

Why it can be worse:

- Needs basis size and penalty choices.
- Can smooth away financial discontinuities in derivatives.
- Does not enforce monotonicity or bounds unless explicitly constrained.

Current role: good future candidate for noisy one-feature labels, especially if
combined with financial constraints.

### Adaptive P-Spline

Motivation: different parts of an option curve need different smoothness.
Deep tails can be nearly flat or linear, while barrier and exercise regions can
change quickly.

Why it can be better:

- Allows more flexibility near kinks and less flexibility in smooth regions.
- Better bias-variance tradeoff than one global smoothing parameter.

Why it can be worse:

- More hyperparameters.
- More risk of overfitting validation noise.
- Harder to explain and audit.

Current role: promising, but only worth the complexity if fixed PCHIP or
ordinary P-spline fails.

### LOESS

Motivation: LOESS fits local low-degree polynomials around each prediction
point.

Why it can be better:

- Good for exploratory smoothing.
- Handles local curvature without committing to global basis functions.
- Can be robustified against outliers.

Why it can be worse:

- Slower at prediction unless carefully cached.
- Boundary behavior can be weak.
- Hard to enforce financial shape constraints.

Current role: diagnostic smoother, not production default.

### Gaussian Process / Matern Smoother

Motivation: a Gaussian process views the unknown price function as a smooth
random function with a covariance kernel.

Why it can be better:

- Provides uncertainty estimates.
- Can denoise labels in a principled way.
- Useful for adaptive sampling: sample more where uncertainty is large.

Why it can be worse:

- Training scales poorly with many states.
- Kernel and length-scale choices matter a lot.
- Pricing can be slower than interpolation or linear regression.

Current role: useful research tool for adaptive design, not the first
production proxy.

### Chebyshev Ridge

Motivation: Chebyshev polynomials are well-conditioned on `[-1, 1]` and are
excellent for smooth functions. Ridge regularization stabilizes noisy MC
labels.

Why it can be better:

- Very fast at pricing time.
- Stable compared with raw monomial regression.
- Denoises because it is a regularized global fit.
- Strong for smooth European-style surfaces.
- Extends naturally to sparse multi-feature bases.

Why it can be worse:

- Global basis can oscillate near kinks or barriers.
- High degree can overfit MC noise.
- Requires good feature scaling.

Current role: best starting point for smooth low-dimensional and
multi-dimensional proxies.

### Piecewise Chebyshev

Motivation: local spectral fits try to keep Chebyshev accuracy while reducing
global oscillation.

Why it can be better:

- More local than one global Chebyshev fit.
- Can handle different smoothness in different regions.
- Useful for products with smooth pieces separated by transition regions.

Why it can be worse:

- Need breakpoints and overlap rules.
- Continuity across pieces must be controlled.
- More moving parts than PCHIP.

Current role: promising bridge between PCHIP and global Chebyshev.

### Bernstein / Bezier Regression

Motivation: Bernstein polynomials have useful shape properties on bounded
intervals. Bezier curves are another representation of polynomial curves and
are natural for shape control.

Why it can be better:

- Can preserve positivity or monotonicity with coefficient constraints.
- Good global approximation for smooth bounded curves.
- Worked well in some European tests.

Why it can be worse:

- Unconstrained Bernstein regression is still a global fit.
- Piecewise Bezier with PCHIP slopes is mathematically just PCHIP in another
  representation.
- Constraints add optimization complexity.

Current role: useful if we explicitly impose financial shape constraints.

### Floater-Hormann Rational Interpolation

Motivation: rational interpolation can approximate some functions better than
polynomials and avoids poles under the Floater-Hormann construction.

Why it can be better:

- Strong approximation for smooth functions with difficult polynomial behavior.
- Barycentric form can be numerically efficient.

Why it can be worse:

- Can still create large excursions.
- Does not automatically respect option-price bounds.
- Performed badly on some barrier log-value tests.

Current role: rejected as a generic default.

## Multi-Feature Methods

### Sparse Chebyshev Ridge

Motivation: full tensor polynomial bases explode in dimension. Sparse
Chebyshev keeps only low-order and payoff-relevant interactions.

Why it can be better:

- Convex linear fit, no neural-network local minima.
- Fast pricing.
- Analytic derivatives are possible.
- Stable when features are scaled.
- Natural extension from the European Chebyshev idea.

Why it can be worse:

- The basis must be chosen carefully.
- Missing interaction terms can create persistent bias.
- Sharp switching surfaces are hard.

Best use: smooth 2D to 10D proxies with well-designed features.

### Baseline Plus Residual

Motivation: do not ask the fitted model to learn the whole price if finance can
explain most of it. Fit a correction to a baseline instead.

Examples:

- Black-Scholes-like baseline for European-style behavior.
- Moment-matched lognormal baseline for basket Asian.
- Intrinsic plus discounted floor/cap baseline for bounded payoffs.

Why it can be better:

- Residual is flatter than raw value.
- Reduces required polynomial degree.
- Helps extrapolation because the baseline carries the main financial shape.

Why it can be worse:

- A bad baseline can introduce bias.
- Ratio targets can misbehave when baseline is too small.

Best use: higher-dimensional products where raw value is too complex.

### PCA Features

Motivation: baskets have many spots, but much of their variation is common
level plus a few dispersion directions.

Why it can be better:

- Reduces dimension while preserving dominant variation.
- Separates basket level from composition.
- Helps with mixed positive and negative correlations.

Why it can be worse:

- PCA is linear and unsupervised; it may miss payoff-important nonlinear
  directions.
- Order-statistic payoffs may care about extremes, not variance explained.

Best use: basket Asian and basket cliquet features, together with level and
payoff-aware variables.

### Neural Networks

Motivation: neural networks are flexible high-dimensional function
approximators.

Why they can be better:

- Scale better than sparse polynomials when dimension and interactions become
  large.
- Can learn nonlinear feature interactions.
- Natural for large training datasets.

Why they can be worse:

- Harder to audit.
- Need architecture and training choices.
- Optimization variability.
- Shape constraints and bounds require extra design.

Current role: future candidate for genuinely high-dimensional exotic proxies,
not the current default.

### Cached Sobol/LR Safety Proxy

Motivation: some hard products are not well represented by one fitted smooth
surface. Basket cliquet order-statistic coupons can switch which asset is best
or worst.

Why it can be better:

- Uses the simulator directly at query states.
- Much safer for discontinuous or switching behavior.
- Can reuse cached Sobol paths and likelihood-ratio mixtures.

Why it can be worse:

- Slower than an analytic fitted proxy.
- Not a pure closed-form surrogate.
- Still has MC error, though much less than a small on-the-fly run.

Current role: safety fallback for basket cliquet hard cases.

## Why The Current Defaults Make Sense

### Why PCHIP Is The One-Feature Default

PCHIP is not always the best pointwise accuracy method. Natural cubic and
Akima can beat it in some cases. It remains the default because it has the best
combination of:

- low implementation risk;
- local behavior;
- no artificial extrema on monotone regions;
- good exercise/barrier behavior;
- fast evaluation;
- easy explanation to a model validator.

For PFE, this is a rational tradeoff. A slightly better average p99 error is
not enough if the method creates fake bumps in exposure-relevant regions.

### Why Chebyshev Is Preferred For Smooth Multi-Feature Fits

Chebyshev ridge gives a fast convex fit with bounded basis columns after
scaling. It is much easier to audit than a neural network and much faster than
a Gaussian process at prediction time. The weakness is basis selection, so the
method should be sparse and feature-aware.

### Why Basket Products Need More Than A Method Name

For basket Asian and basket cliquet, the main question is not "PCHIP or
Chebyshev?" The main question is feature design:

- basket level;
- running sum or accrued coupon;
- volatility state;
- correlation and dispersion;
- PCA scores;
- floor/cap cushion;
- worst/best ranking indicators or approximations.

Once the features are good, the fitting method has a simpler job.

## Recommended Method Ladder

Use this ladder before inventing a new method.

1. Verify the state is pricing-sufficient.
2. Add exact payoff regions: maturity, intrinsic value, barrier hit status,
   floor/cap tails, in/out parity.
3. Improve MC labels: Sobol, antithetics, control variates, common random
   numbers, LR mixture importance sampling.
4. Choose a financially scaled coordinate: `d1`, adjusted moneyness, accrued
   cushion, normalized variance, PCA score.
5. Try the simplest safe proxy:
   - PCHIP for one feature;
   - sparse Chebyshev ridge for several smooth features.
6. If labels are noisy, test smoothing spline or P-spline.
7. If the target has local slope changes, test Akima or MAKIMA.
8. If the target is globally smooth, test Chebyshev or Bernstein ridge.
9. If the product is high-dimensional, add baseline plus residual and PCA.
10. If strict max error still fails, use adaptive sampling or a cached Sobol/LR
    safety proxy in the hard region.

## Improvement Backlog

These are reasonable next improvements.

### 1. Adaptive State Sampling

Use validation error to add more training states where the signed relative
error is large. This is better than blindly increasing every grid size.

Priority products:

- barriers near barrier levels;
- Bermudan near exercise boundaries;
- basket cliquet near best/worst switching surfaces.

### 2. State-Dependent Path Allocation

Allocate more MC paths to high-variance or high-relative-error states. Low-noise
states should not consume the same budget as difficult transition states.

### 3. Shape-Constrained Fits

Test constrained splines or constrained Bernstein fits when finance gives a
known shape:

- monotone call versus spot;
- convex European call versus spot;
- bounded cliquet price;
- American value above intrinsic.

This may improve validator confidence even when raw accuracy is similar.

### 4. Adaptive Sparse Chebyshev

Start with low-order terms, then add interaction terms only when they reduce
out-of-sample error. This directly addresses the biggest weakness of
multi-feature Chebyshev: basis selection.

### 5. Mixture Importance Sampling By Product

Use product-aware shift components:

- lower-tail shifts for put, down barrier, and worst-of;
- upper-tail shifts for call, up barrier, and best-of;
- dispersion shifts for basket order statistics;
- central component always included for stability.

### 6. Outer-Scenario Weighted Validation

For PFE, validation states should be weighted by the outer scenario
distribution and positive-exposure region. Pointwise uniform grids are useful,
but they are not the final PFE objective.

### 7. Neural Network Benchmark For High Dimension

For 10D and above, train a small smooth neural network as a benchmark against
sparse Chebyshev. Use monotone or bounded output transforms when available.

This is not a replacement default yet. It is a research benchmark for the
dimension where sparse polynomial basis selection becomes hard.

## Current Practical Recommendation

- Keep PCHIP as the generic one-feature default.
- Use Akima as the first local challenger.
- Use Chebyshev or Bernstein ridge when the one-feature target is globally
  smooth and labels contain MC noise.
- Use smoothing spline or P-spline when label noise is the obvious problem.
- Use sparse Chebyshev with baseline and PCA for higher-dimensional products.
- For basket cliquet order-statistic cases, keep the cached Sobol/LR safety
  proxy until a fitted method proves it can control worst-case error.

