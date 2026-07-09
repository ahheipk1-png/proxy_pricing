# European Option Proxy Findings

This note summarizes what we learned from the European call option proxy experiments.
The benchmark product was a 1D European call under GBM, so Black-Scholes was used only
as an accuracy benchmark. The proxy-training experiments are meant to guide a more
general Monte Carlo workflow for exotic products.

## Main Conclusion

The strongest finding is:

> For max percentage-error control, the training state distribution matters more than
> the regression basis.

When the proxy was trained only on paths starting from `S0 = 100`, it fit the central
region well but behaved badly in the early-time wings. This happened because the
training paths rarely visited those wing states. Adding asymptotic anchors helped, but
the real improvement came from deliberately training on the full state domain where we
wanted accuracy.

The best generic workflow is:

```text
choose the pricing domain
sample states across that domain, including wings
use tail-biased / shifted MC to estimate labels at those states
fit log(value), not raw value
validate max relative error on a holdout grid
```

## Universal One-Dimensional Default

The standalone default was changed to log-PCHIP on the same `d1` coordinate so
European, Asian, and American one-dimensional proxies share one shape-preserving
fitting engine. With an independent default seed it produced:

| Time | Max % Error | P99 % Error | MAE |
|---:|---:|---:|---:|
| 0.20 | 1.375% | 0.932% | 0.013603 |
| 0.40 | 2.637% | 1.577% | 0.011506 |
| 0.60 | 1.379% | 1.251% | 0.007895 |
| 0.80 | 2.080% | 1.671% | 0.005021 |
| 1.00 | 0.000% | 0.000% | 0.000000 |

The historical Chebyshev and Bernstein experiments remain more accurate for
this globally smooth product. PCHIP is the default because it provides one
local, monotonicity-preserving method across all genuinely one-dimensional
products.

## Best Product-Specific Method

The best practical method for this European test was:

```text
state grid:        121 points across delta 0.0001 to 0.9999
MC labels:         shifted / importance-sampled MC
paths per state:   25,000
target:            log(option value)
fitters tested:    polynomial, Chebyshev, B-spline, LOESS, PCHIP, Fourier
universal fitter:  log PCHIP in d1 coordinate
optimized fitter:  log Chebyshev or Bernstein ridge
diagnostic metric: abs(proxy - truth) / max(truth, 0.01)
```

The normal samples were shifted in the wings:

```text
Y = Z + shift
```

and reweighted by the likelihood ratio:

```text
weight = exp(-shift * Y + 0.5 * shift^2)
```

This keeps the Monte Carlo label unbiased while forcing more samples into payoff-relevant
wing scenarios.

## Key Results

### Tail-biased MC only, no asymptotic proxy

This version used no asymptotic proxy, no asymptotic anchors, no asymptotic mixing, and
no asymptotic clipping.

The default plotted method is now `log Chebyshev | d1, degree=7`.

| Time | Max % Error | P99 % Error | MAE |
|---:|---:|---:|---:|
| 0.20 | 0.227% | 0.146% | 0.012845 |
| 0.40 | 0.183% | 0.183% | 0.004373 |
| 0.60 | 0.089% | 0.085% | 0.002880 |
| 0.80 | 0.370% | 0.370% | 0.002398 |
| 1.00 | 0.000% | 0.000% | 0.000000 |

This comfortably beat the original 3-5% max-error target.

### Best methods in the no-asym bakeoff

| Method | Worst Max % Error | Avg P99 % Error | Avg MAE |
|---|---:|---:|---:|
| log polynomial, spot degree 11 | 0.359% | 0.205% | 0.004184 |
| log polynomial, spot degree 9 | 0.359% | 0.193% | 0.004643 |
| log Chebyshev, d1 degree 7 | 0.370% | 0.196% | 0.005624 |
| log Chebyshev, logm degree 7 | 0.370% | 0.196% | 0.005624 |
| log B-spline, d1 knots 12 | 0.447% | 0.341% | 0.004482 |

Raw high-degree polynomial happened to score slightly better in this 1D European test,
but the default should be `log Chebyshev | d1, degree=7`. It was nearly as accurate in
the single-option test and was the most robust method in the broader option grid sweep.

## What We Learned By Stage

### 1. Ordinary path-trained regression was not enough

Training only on paths from `S0 = 100` gave decent average error but poor max percentage
error in early-time wings. The proxy was being asked to extrapolate in regions where it
had little or no training information.

This explains the large relative-error spikes in the OTM / transition regions.

### 2. Asymptotic tail anchors helped, but did not solve the main issue

Adding synthetic asymptotic tail points improved wide-range behavior.

The best direct tail-anchor version improved wide-grid MAE approximately from:

```text
baseline B-spline wide-grid MAE:      0.03372
tail-anchored B-spline wide-grid MAE: 0.01129
```

But this still relied on a handoff between learned proxy and asymptotic proxy. It helped
because the training data was missing in the wings; it was not the fundamental solution.

### 3. Shifted MC labels fixed the data problem

Once we deliberately sampled states across the full target delta range and used shifted
MC to estimate wing labels, the asymptotic proxy was no longer needed for this European
test.

Plain MC labels were much less efficient in the wings. In one test:

| Labeling method | Worst Max % Error |
|---|---:|
| plain MC, 50k paths/state | 16.07% |
| shifted MC, 10k paths/state | 0.98% |
| shifted MC, 25k paths/state | 0.51% |
| shifted MC, 50k paths/state | 0.56% |
| shifted MC, 100k paths/state | 0.25% |

The important point is not the exact number, but the pattern: shifted MC gave much better
wing labels for the same or lower simulation budget.

### 4. Fitting log(value) was consistently better for percentage error

Fitting raw values tends to prioritize the large ITM region. Fitting `log(value)` makes
small option values matter more, which is exactly what we want when controlling relative
error.

For near-zero prices, percentage error was computed with a floor:

```text
relative_error = abs(proxy - truth) / max(truth, 0.01)
```

Without this floor, percentage error is not meaningful when the true value is almost zero.

## Coordinate Choices

We tested several coordinates:

| Name | Meaning |
|---|---|
| `spot` | normalized spot, roughly `S / S0 - 1` |
| `logm` | log-moneyness, `log(S / K)` |
| `d1` | Black-Scholes-style standardized moneyness |

For a European option, `d1` and `logm` are often smoother coordinates than raw spot.
For more exotic products, the right coordinates should be chosen from the state variables
that naturally describe the payoff, such as spot, running average, barrier distance,
basket weights, realized variance, or time.

## Basis Comparison

### Chebyshev

Chebyshev regression is worth keeping. It is essentially polynomial regression in a
better-conditioned basis. In the 1D test it did not materially outperform raw polynomial,
but it is more robust and should scale better as dimension rises.

Recommended higher-dimensional direction:

```text
adaptive sparse Chebyshev regression
```

That means:

```text
start with low-degree terms
add interaction terms only when validation error improves
regularize with ridge or elastic net
stop when holdout max % error stops improving
```

### B-spline

B-spline worked well in 1D and is a good secondary smoother for low-dimensional
problems. However, tensor-product splines become expensive in higher dimensions, so it
is no longer the default.

Use B-spline for:

```text
1D to 2D, maybe 3D
```

For 5D to 10D, sparse Chebyshev or neural networks are more natural.

### Fourier

Fourier was not suitable here. Option values are not periodic over the spot interval, so
Fourier bases caused boundary ringing. The best Fourier result was far worse than the
other methods.

Recommendation:

```text
avoid Fourier basis for this proxy problem unless the target is naturally periodic
```

## Recommendation For Higher-Dimensional Exotics

For exotic products, the recommended research path is:

```text
1. define the state variables
2. sample the full state domain, not just today-to-maturity paths
3. oversample wings / rare-event regions
4. use shifted MC with likelihood-ratio reweighting
5. fit log(value)
6. compare sparse Chebyshev, B-spline if dimension is small, and neural nets
7. select by holdout max % error, not only average error
```

My preferred next experiment is a 2D or 3D exotic proxy, for example:

```text
Asian option state: spot, running average, time
```

Then compare:

```text
log sparse Chebyshev (default)
log B-spline
log polynomial
small neural network
```

using the same tail-biased MC label-generation methodology.

## Current Best Finding

The best finding is:

> The asymptotic proxy is optional if we train the wings properly. Tail-biased MC state
> sampling plus log-value regression controls max percentage error far better than
> relying on ordinary path samples plus asymptotic patches.

For this European benchmark, the no-asymptotic tail-biased MC proxy achieved sub-1% max
relative error across the tested time slices.
