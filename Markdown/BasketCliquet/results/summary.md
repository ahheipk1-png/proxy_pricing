# Generalized three-underlying SLV basket cliquet experiment

The payoff cases are based on the generalized multi-asset cliquet note:
weighted-average, basket-ratio, order-statistic, and spread/bonus local coupons.
All cases use sum aggregation and the same global floor/cap payoff.

Training uses 1,009 low-discrepancy market states, 17 accrued-return layers per market state, and grouped Sobol/LR labels so one simulated future-coupon distribution prices all accrued layers for that market state.

The SLV path sampler uses antithetic Sobol points and an 8-component likelihood-ratio mixture over common market and dispersion directions. The proxy feature set includes lower/upper payoff cushions and PCA coordinates of log spots and log variances.

Conclusion: this is a useful improvement for basket-like coupons, but not yet a universal 5-8% method for all generalized basket cliquets. Order-statistic coupons remain the hard cases.

| Variant | Best method | Worst max relative error | Average p99 | Average MAE |
|---|---|---:|---:|---:|
| `basket_return` | `sobol_mc_proxy` | 2.519% | 1.179% | 0.005435 |
| `weighted_average` | `sobol_mc_proxy` | 4.053% | 1.458% | 0.006072 |
| `basket_ratio` | `sobol_mc_proxy` | 1.265% | 0.865% | 0.005393 |
| `average_clipped` | `sobol_mc_proxy` | 1.339% | 0.653% | 0.004645 |
| `second_worst` | `sobol_mc_proxy` | 2.853% | 1.218% | 0.005230 |
| `worst_of` | `sobol_mc_proxy` | 2.206% | 1.436% | 0.004951 |
| `best_of` | `sobol_mc_proxy` | 3.896% | 1.124% | 0.008814 |
| `spread_bonus` | `sobol_mc_proxy` | 3.242% | 2.618% | 0.004328 |
