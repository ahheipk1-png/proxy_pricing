# One-dimensional fitting comparison

| Product | Method | Worst max relative error | Average p99 | Average MAE |
|---|---|---:|---:|---:|
| european | `chebyshev` | 0.854% | 0.326% | 0.004423 |
| european | `pchip` | 2.554% | 1.149% | 0.007098 |
| european | `akima` | 2.540% | 1.127% | 0.007015 |
| european | `bezier_global` | 0.536% | 0.208% | 0.003546 |
| european | `bezier_pchip` | 2.554% | 1.149% | 0.007098 |
| asian | `chebyshev` | 3.286% | 1.721% | 0.013523 |
| asian | `pchip` | 3.453% | 0.936% | 0.000273 |
| asian | `akima` | 0.186% | 0.057% | 0.000178 |
| asian | `bezier_global` | 6.854% | 3.303% | 0.017907 |
| asian | `bezier_pchip` | 3.453% | 0.936% | 0.000273 |
