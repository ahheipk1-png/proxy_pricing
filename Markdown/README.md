# Proxy Pricing

Small research prototype for Monte Carlo-trained option-pricing proxies.

Future agents should start with `START_HERE.md`, then read the option-specific
folders such as `European/`, `Asian/`, `American/`, `Barrier/`,
`BasketAsian/`, and `BasketCliquet/`.

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
python AutocallableMain.py
python BermudanMain.py
python RandomOptionMain.py
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
generalized multi-asset cliquet payoff cases
weighted-average, basket-ratio, order-statistic, and spread/bonus coupons
state = accrued return, 3 spots, and 3 variances
grouped Sobol/LR labels across accrued-return layers
8-component common/dispersion likelihood-ratio mixture
PCA coordinates for log spots and log variances
local, sparse Chebyshev, anchored residual, accrued-PCHIP/kNN comparisons
cached 65,536-path Sobol/LR safety proxy for hard generalized coupons
```

The fitted-only basket cliquet result is mixed: grouped labels and PCA features
bring `basket_return` to 6.448% worst max error and `average_clipped` to
7.967%, but some order-statistic coupons remain above target. The cached
`sobol_mc_proxy` safety layer was added for those hard cases; in a 31-state
month 3/6/9 spot check against 262,144-path benchmarks, the hard-case worst max
errors were 3.0% to 11.2%.

`OneDimensionalFitExperiment/ExpandedSplineStudy.py` compares 17 spline,
local, spectral, rational, and kernel methods across 99 European, American,
Asian, and barrier cases. PCHIP remains the generic one-feature default because
it was essentially tied for best balanced accuracy and had zero local
overshoot.

`AutocallableMain.py` runs a single-underlying autocallable proxy grid with
five payoff/market cases and four observation dates.

`BermudanMain.py` runs a monthly Bermudan put proxy grid with five parameter
cases. Labels use Sobol/LR mixture MC dynamic programming and benchmarks use an
independent Bermudan tree.

`RandomOptionMain.py` runs a fixed-seed random piecewise-linear terminal-payoff
stress test across eight payoff shapes and three market regimes.

Exploratory scripts, plots, and CSVs are stored in:

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

Markdown notes and generated markdown summaries are centralized by option type:

```text
Markdown/European/
Markdown/Asian/
Markdown/American/
Markdown/Barrier/
Markdown/Cliquet/
Markdown/SLVCliquet/
Markdown/BasketAsian/
Markdown/BasketCliquet/
Markdown/Autocallable/
Markdown/Bermudan/
Markdown/RandomOption/
Markdown/MethodStudy/
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
AutocallableOptExperiment/results/autocallable_proxy_method_results.csv
BermudanOptExperiment/results/bermudan_proxy_method_results.csv
RandomOptionOptExperiment/results/random_option_proxy_method_results.csv
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Correctness Tests

Run the lightweight invariant suite with:

```powershell
python ProxyCorrectnessTests.py
```

These tests cover European, Asian, American, barrier, cliquet, SLV cliquet,
basket Asian, basket cliquet, autocallable, Bermudan, and random payoff cases.

Generate the representative timing/performance table with:

```powershell
python ProxyTimingBenchmark.py
```

The timing benchmark uses multiple worker processes by default. Override the
worker count with:

```powershell
$env:PROXY_BENCHMARK_WORKERS = "8"
python ProxyTimingBenchmark.py
```

For clean sequential option-family timing, set the worker count to one:

```powershell
$env:PROXY_BENCHMARK_WORKERS = "1"
python ProxyTimingBenchmark.py
```

Generate the expanded coverage report, which asserts at least 100 cases per
option type, with:

```powershell
python ProxyExpandedCoverageTests.py
```

## Methodology Guide

The detailed mathematical and implementation guide is:

```text
output/pdf/proxy_pricing_methodology.pdf
```

The guide is written as a LaTeX textbook for a second-year undergraduate with
calculus, probability, and linear algebra. Its 63 physical pages start with
motivation, solution overview, diagnostic graphs, and guided examples before
moving into the technical chapters. It includes:

```text
risk-system motivation and the repeated-pricing problem
solution workflow graph and representative value/error diagnostic plots
guided examples for European, Asian, barrier, American, cliquet, and basket products
risk-neutral valuation, state sufficiency, and error decomposition
Sobol path generation, antithetics, control variates, and LR mixture IS
step-by-step proxy recipes and implementation algorithms for every instrument
complete PCHIP and sparse Chebyshev construction details
line-by-line proof appendices for the main formulas and theorem statements
primary academic references
```

It can be regenerated with:

```powershell
python Documentation/generate_proxy_methodology_pdf.py
```

The build uses Tectonic. If `tectonic` is not on `PATH`, place `tectonic.exe`
at `tools/tectonic/tectonic.exe`; the local compiler binary is intentionally
ignored by Git.
