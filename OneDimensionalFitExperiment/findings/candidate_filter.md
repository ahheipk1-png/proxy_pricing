# Candidate filter for a generic one-feature option proxy

The comparison uses distinct numerical estimators, not every name that can be
attached to the same estimator.

## Included

| Experimental method | Names represented |
|---|---|
| PCHIP | Shape-preserving piecewise cubic Hermite interpolation |
| Akima | Local slope-weighted cubic interpolation |
| MAKIMA | Modified Akima with additional slope weights |
| Linear interpolation | Dense-grid, no-overshoot production baseline |
| Natural cubic interpolation | Natural cubic spline with all observations as knots |
| B-spline regression | Polynomial regression spline, truncated-power spline in a better-conditioned basis, cubic regression-spline GAM smooth |
| Natural cubic regression | Natural/restricted cubic regression spline |
| Cubic smoothing spline | Classical smoothing spline and the one-dimensional thin-plate analogue |
| P-spline | Penalized spline, penalized cubic regression spline, P-spline GAM smooth, Gaussian-prior/Bayesian posterior mode for fixed smoothing |
| Adaptive P-spline | Adaptive penalty/adaptive GAM smooth |
| Adaptive-knot B-spline | Practical free-knot representative |
| LOESS | Local polynomial smooth regression |
| Matern GP | Gaussian-process smoother and GP smooth inside a GAM |
| Chebyshev ridge | Stable global polynomial reference |
| Piecewise Chebyshev | Overlapping low-degree local spectral fits |
| Bernstein ridge | Global Bernstein/Bezier reference |
| Floater-Hormann | Pole-free barycentric rational interpolation |

## Excluded from the universal contest

- Cyclic cubic smooths assume periodic endpoints; option value versus state is
  not periodic.
- Tensor products, thin plates in multiple dimensions, Duchon smooths, and
  soap-film smoothers solve multivariate or irregular-domain problems.
- GAMM and varying-coefficient models add grouping or another predictor, not a
  new one-dimensional curve estimator.
- Shrinkage smooths decide whether an additive term should vanish. A pricing
  proxy has one required state term.
- M-, I-, C-, and SCAM splines are valuable when monotonicity or convexity is
  known. They are not a universal default because barrier prices can be
  non-monotone and early-exercise or transformed time-value curves can contain
  kinks.
- Bayesian splines are not a separate curve family until a prior and
  uncertainty objective are specified. For Gaussian observations, their
  posterior mode is the penalized-spline fit already tested here.
- Restricted cubic spline is another name for the natural cubic regression
  family in this setting.

## Selection rule

Hyperparameters are selected from fixed candidate grids using GCV or
deterministic five-fold cross-validation on training labels only. Final ranking
uses independent option values and gives equal weight to each product/parameter
case so that one large test family cannot dominate the conclusion.
