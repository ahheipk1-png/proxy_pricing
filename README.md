# Proxy Pricing

Small research prototype for Monte Carlo-trained option-pricing proxies.

The current parent-level entry points are:

```powershell
python EuroMain.py
python AsianMain.py
python CliquetMain.py
```

`EuroMain.py` runs only the current European option default:

```text
tail-biased shifted MC labels
log(option value) target
Chebyshev regression in d1
degree 7
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
Chebyshev degree 19
shifted antithetic MC labels
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

`CliquteMain.py` is a compatibility wrapper for the common misspelling and calls
`CliquetMain.py`.

All exploratory European, Asian, and cliquet option scripts, plots, CSVs, and
notes are stored in:

```text
EuroOptExperiment/
AsianOptExperiment/
CliquetOptExperiment/
```

The default runs write:

```text
EuroOptExperiment/default_run/euro_default_results.csv
AsianOptExperiment/default_run/asian_default_results.csv
CliquetOptExperiment/default_run/cliquet_default_results.csv
```

## Setup

```powershell
python -m pip install -r requirements.txt
```
