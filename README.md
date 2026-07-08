# Proxy Pricing

Small research prototype for Monte Carlo-trained option-pricing proxies.

The current parent-level entry points are:

```powershell
python EuroMain.py
python AsianMain.py
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

All exploratory European and Asian option scripts, plots, CSVs, and notes are
stored in:

```text
EuroOptExperiment/
AsianOptExperiment/
```

The default runs write:

```text
EuroOptExperiment/default_run/euro_default_results.csv
AsianOptExperiment/default_run/asian_default_results.csv
```

## Setup

```powershell
python -m pip install -r requirements.txt
```
