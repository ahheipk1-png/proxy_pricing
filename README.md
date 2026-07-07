# Proxy Pricing

Small research prototype for regression-based option pricing proxies.

The current script builds a Stage 1 proxy for European option values under
geometric Brownian motion. It simulates risk-neutral paths, fits polynomial
least-squares proxies at each time point, and compares the fitted proxy against
Black-Scholes values on representative spot grids.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python stage1_lsmc_european.py
```

## Current Scope

- Exact GBM path simulation under the risk-neutral measure
- European call/put payoff support
- Polynomial basis regression with an additional payoff basis term
- Black-Scholes validation table and fitted coefficient output

