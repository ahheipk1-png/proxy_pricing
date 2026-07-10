# Proxy Pricing

Small research prototype for Monte Carlo-trained option-pricing proxies.

The current parent-level entry points are:

```powershell
python EuroMain.py
python AsianMain.py
python CliquetMain.py
python SLVCliquetMain.py
python AmericanMain.py
python BarrierMain.py
python BasketAsianMain.py
python BasketCliquetMain.py
python OneDimensionalFitExperiment/ExpandedSplineStudy.py
```

`EuroMain.py` runs only the current European option default:

```text
Sobol tail-biased shifted MC labels
log(option value) target
SciPy PCHIP interpolation in d1
shape-preserving cubic Hermite slopes
no asymptotic proxy
```

It is standalone and does not import the exploratory modules under
`EuroOptExperiment/`.

`AsianMain.py` runs only the current Asian option default:

```text
monthly arithmetic Asian call
12 observations over 1 year
adjusted-moneyness state transform
log(value / spot) and log(time value / spot) targets
SciPy PCHIP interpolation for both one-dimensional targets
Sobol shifted antithetic MC labels
geometric Asian control variate for benchmark
```

It is standalone and does not import the exploratory modules under
`AsianOptExperiment/`.

`CliquetMain.py` runs only the current cliquet option default:

```text
monthly cliquet
12 observations over 1 year
local floor/cap = -2% / 4%
global floor/cap = 0% / 20%
expected-total z feature
bounded-logit value target
Chebyshev degree 19
boundary-enriched accrued-return training grid
```

It is standalone and does not import the exploratory modules under
`CliquetOptExperiment/`.

`SLVCliquetMain.py` runs the three-state SLV cliquet default:

```text
state = accrued return, spot, variance
bounded-logit price target
Sobol lower-tail likelihood-ratio importance sampling
local quadratic fit with 9-12 resets remaining
anisotropic Chebyshev fit with 3-6 resets remaining
```

`AmericanMain.py` runs the American put default:

```text
100-date MC dynamic programming
Sobol one-step training transitions
SciPy PCHIP continuation proxy
exact intrinsic-value exercise constraint
projected finite-difference benchmark
```

`BarrierMain.py` runs the barrier study:

```text
down, up, and double knock-out calls
monthly discrete and continuous monitoring
Brownian-bridge continuous-crossing correction
common-Sobol and antithetic MC labels
SciPy PCHIP, SciPy Akima, Chebyshev, and Bernstein comparison
```

Zero-rebate knock-ins are obtained by in/out parity. The barrier study uses
at least 10 million training scenarios per fitted date and 524,288 benchmark paths per
validation state.

`BasketAsianMain.py` runs the 10-underlying basket Asian study:

```text
monthly arithmetic Asian call on an equal-weight 10-asset basket
mixed positive and negative correlations
state = 10 spots plus running sum of previous basket fixings
Sobol MC labels with conditional likelihood-ratio mixture importance sampling
524,288-path Sobol/LR benchmarks per validation state
moment-matched lognormal baseline
PCA sparse-Chebyshev log-factor correction
PCHIP residual calibration as the default
```

It is standalone and does not import exploratory modules from any subfolder.
The current default method is `pchip_calibrated_log_factor_pca`.

`BasketCliquetMain.py` runs the three-underlying SLV basket cliquet study:

```text
3 underlying SLV state variables plus accrued return
Sobol batched MC with antithetic and shifted mixture samples
sparse Chebyshev, local quadratic, and adaptive blend comparisons
```

`OneDimensionalFitExperiment/ExpandedSplineStudy.py` compares 17 spline,
local, spectral, rational, and kernel methods across 99 European, American,
Asian, and barrier cases. PCHIP remains the generic one-feature default because
it was essentially tied for best balanced accuracy and had zero local
overshoot.

All exploratory scripts, plots, CSVs, and notes are stored in:

```text
EuroOptExperiment/
AsianOptExperiment/
CliquetOptExperiment/
SLVCliquetOptExperiment/
AmericanOptExperiment/
BarrierOptExperiment/
BasketAsianOptExperiment/
BasketCliquetOptExperiment/
OneDimensionalFitExperiment/
```

The default runs write:

```text
EuroOptExperiment/default_run/euro_default_results.csv
AsianOptExperiment/default_run/asian_default_results.csv
CliquetOptExperiment/default_run/cliquet_default_results.csv
SLVCliquetOptExperiment/default_run/slv_cliquet_default_results.csv
AmericanOptExperiment/default_run/american_default_results.csv
BarrierOptExperiment/results/barrier_proxy_method_results.csv
BasketAsianOptExperiment/results/basket_asian_proxy_method_results.csv
BasketCliquetOptExperiment/results/basket_slv_cliquet_proxy_method_results.csv
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Methodology Guide

The detailed mathematical and implementation guide is:

```text
output/pdf/proxy_pricing_methodology.pdf
```

The guide is written for a second-year undergraduate with calculus,
probability, and linear algebra. Its 42 pages include:

```text
risk-neutral pricing and Monte Carlo label construction
step-by-step proxy recipes for every instrument
complete PCHIP and sparse Chebyshev algorithms
theorem statements, assumptions, proofs or proof sketches
primary academic references
```

It can be regenerated with:

```powershell
python Documentation/generate_proxy_methodology_pdf.py
```
