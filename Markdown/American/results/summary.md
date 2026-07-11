# American put proxy experiment

Best method: `pchip_spline`.

| Method | Worst max relative error | Average p99 | Average MAE |
|---|---:|---:|---:|
| `direct_chebyshev_d9` | 8695.717% | 6194.163% | 0.360034 |
| `linear_spline` | 126.023% | 110.024% | 0.096668 |
| `pchip_spline` | 2.249% | 0.947% | 0.001479 |
| `akima_spline` | 3.225% | 1.675% | 0.002033 |
| `log_chebyshev_d7` | 5541.197% | 4423.119% | 148.866438 |
