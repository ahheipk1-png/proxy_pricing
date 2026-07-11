# Correctness Test Suite

Run:

```powershell
python ProxyCorrectnessTests.py
```

The suite is intentionally lightweight. It does not replace the heavy MC
performance grids; it checks identities and invariants that should hold every
time the pricing/proxy code changes.

## Coverage

| Option type | Checks |
|---|---|
| European | Put-call parity, terminal payoff, `d1` coordinate round trip. |
| Asian | Terminal payoff and the one-future-fixing identity against European Black-Scholes. |
| American | Small MC dynamic-programming proxy is finite and never below intrinsic value. |
| Barrier | Brownian-bridge survival probabilities stay in `[0, 1]`; in/out parity is exact. |
| Cliquet | Locked global floor/cap tails and payoff bounds. |
| SLV cliquet | Local-vol leverage bounds and locked tail values. |
| Basket Asian | Correlation matrix validity, PCA orthonormality, terminal payoff. |
| Basket cliquet | PCA orthonormality, coupon bounds, locked tails for order-statistic coupon. |
| Autocallable | Deterministic maturity redemption by payoff region. |
| Bermudan | Tree maturity equals intrinsic and early value is at least intrinsic. |
| Random payoff | Zero-time MC value equals terminal payoff; interpolation is exact at payoff knots. |

## Bug Found

The first run caught a real basket Asian issue: the correlation construction
normalized to a correlation matrix and then clipped the full matrix, including
the diagonal. That changed the diagonal from `1.0` to `0.95`, which is not a
valid correlation matrix. `BasketAsianMain.correlation_matrix` now restores the
diagonal to `1.0` after clipping off-diagonal entries.
