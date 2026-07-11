# Expanded one-dimensional smoother comparison

Methods are ranked by the mean product-family p99 relative error. Each
product family receives equal weight. Hyperparameters use training labels only.

| Rank | Method | Balanced avg p99 | Avg max >= $0.05 | Local overshoot | Raw worst |
|---:|---|---:|---:|---:|---:|
| 1 | `natural_cubic_interpolation` | 1.799% | 1.163% | 2.01% | 19.706% |
| 2 | `pchip` | 1.834% | 1.220% | 0.00% | 19.702% |
| 3 | `akima` | 1.958% | 1.286% | 0.17% | 19.721% |
| 4 | `makima` | 1.968% | 1.317% | 0.15% | 19.703% |
| 5 | `cubic_smoothing_spline` | 2.098% | 1.440% | 2.49% | 37.836% |
| 6 | `linear` | 2.683% | 2.058% | 0.00% | 28.972% |
| 7 | `matern_gp` | 5.121% | 4.989% | 6.89% | 51.318% |
| 8 | `bspline_regression` | 8.808% | 5.794% | 15.59% | 181.931% |
| 9 | `pspline` | 10.557% | 6.804% | 15.09% | 254.310% |
| 10 | `loess` | 11.051% | 8.747% | 8.49% | 100.000% |
| 11 | `adaptive_pspline` | 11.084% | 6.578% | 16.33% | 270.282% |
| 12 | `free_knot_spline` | 19.831% | 7.783% | 15.46% | 283.868% |
| 13 | `natural_cubic_regression` | 56.448% | 15.180% | 22.62% | 983.686% |
| 14 | `chebyshev` | 74.119% | 12.261% | 25.87% | 1610.378% |
| 15 | `bernstein` | 94.448% | 13.166% | 24.92% | 1953.380% |
| 16 | `piecewise_chebyshev` | 148.699% | 15.136% | 22.75% | 2616.459% |
| 17 | `floater_hormann` | 11318976354.774% | 10064270531.531% | 21.53% | 1083893622033.586% |

## Product matrix

| Product | Method | Avg p99 | Avg max >= $0.05 | Worst max | Avg MAE |
|---|---|---:|---:|---:|---:|
| american | `natural_cubic_interpolation` | 0.336% | 0.251% | 0.769% | 0.000285 |
| american | `cubic_smoothing_spline` | 0.341% | 0.265% | 0.754% | 0.000287 |
| american | `matern_gp` | 0.342% | 0.253% | 0.773% | 0.000285 |
| american | `floater_hormann` | 0.455% | 0.253% | 1.092% | 0.000287 |
| american | `pchip` | 0.610% | 0.513% | 4.157% | 0.000322 |
| american | `akima` | 1.055% | 0.777% | 8.948% | 0.000334 |
| american | `makima` | 1.109% | 0.900% | 9.285% | 0.000349 |
| american | `linear` | 3.790% | 3.685% | 28.972% | 0.001858 |
| american | `bspline_regression` | 12.512% | 2.280% | 181.931% | 0.000572 |
| american | `pspline` | 19.818% | 5.161% | 254.310% | 0.000733 |
| american | `adaptive_pspline` | 21.369% | 4.986% | 270.282% | 0.000738 |
| american | `loess` | 26.578% | 18.878% | 100.000% | 0.002553 |
| american | `free_knot_spline` | 68.552% | 23.629% | 283.868% | 0.002528 |
| american | `natural_cubic_regression` | 183.003% | 26.824% | 983.686% | 0.007933 |
| american | `chebyshev` | 271.387% | 24.268% | 1610.378% | 0.008429 |
| american | `bernstein` | 353.260% | 29.703% | 1953.380% | 0.011296 |
| american | `piecewise_chebyshev` | 563.083% | 31.745% | 2616.459% | 0.017383 |
| asian | `makima` | 0.066% | 0.041% | 0.224% | 0.000894 |
| asian | `pchip` | 0.066% | 0.041% | 0.224% | 0.000893 |
| asian | `matern_gp` | 0.066% | 0.041% | 0.228% | 0.000886 |
| asian | `natural_cubic_interpolation` | 0.066% | 0.041% | 0.228% | 0.000886 |
| asian | `akima` | 0.066% | 0.041% | 0.231% | 0.000894 |
| asian | `cubic_smoothing_spline` | 0.066% | 0.041% | 0.228% | 0.000886 |
| asian | `floater_hormann` | 0.073% | 0.050% | 0.242% | 0.000989 |
| asian | `bspline_regression` | 0.134% | 0.137% | 0.866% | 0.001484 |
| asian | `linear` | 0.181% | 0.153% | 0.344% | 0.001038 |
| asian | `free_knot_spline` | 0.191% | 0.172% | 0.683% | 0.001442 |
| asian | `adaptive_pspline` | 0.203% | 0.222% | 1.096% | 0.001858 |
| asian | `pspline` | 0.205% | 0.231% | 1.113% | 0.001866 |
| asian | `loess` | 0.235% | 0.340% | 1.547% | 0.002061 |
| asian | `natural_cubic_regression` | 1.201% | 1.108% | 2.495% | 0.005937 |
| asian | `chebyshev` | 1.582% | 1.706% | 3.488% | 0.009355 |
| asian | `bernstein` | 2.069% | 2.064% | 3.464% | 0.010436 |
| asian | `piecewise_chebyshev` | 2.099% | 1.729% | 5.284% | 0.005862 |
| barrier | `pchip` | 6.248% | 3.961% | 19.702% | 0.018963 |
| barrier | `makima` | 6.286% | 3.962% | 19.703% | 0.018971 |
| barrier | `akima` | 6.300% | 3.961% | 19.721% | 0.018974 |
| barrier | `linear` | 6.348% | 4.020% | 19.734% | 0.018995 |
| barrier | `natural_cubic_interpolation` | 6.383% | 3.994% | 19.706% | 0.018967 |
| barrier | `cubic_smoothing_spline` | 7.575% | 5.087% | 37.836% | 0.019743 |
| barrier | `free_knot_spline` | 10.168% | 6.964% | 83.084% | 0.034648 |
| barrier | `loess` | 16.981% | 15.405% | 34.845% | 0.021143 |
| barrier | `matern_gp` | 19.664% | 19.297% | 51.318% | 0.024154 |
| barrier | `pspline` | 21.789% | 21.456% | 48.373% | 0.033913 |
| barrier | `bernstein` | 22.049% | 20.531% | 55.466% | 0.138213 |
| barrier | `bspline_regression` | 22.174% | 20.392% | 46.834% | 0.047280 |
| barrier | `adaptive_pspline` | 22.351% | 20.738% | 44.866% | 0.041791 |
| barrier | `chebyshev` | 23.095% | 22.704% | 69.417% | 0.239592 |
| barrier | `piecewise_chebyshev` | 29.204% | 26.703% | 66.184% | 0.087392 |
| barrier | `natural_cubic_regression` | 41.176% | 32.422% | 100.318% | 0.100491 |
| barrier | `floater_hormann` | 45275905418.157% | 40257082125.455% | 1083893622033.586% | 11872689.045629 |
| european | `piecewise_chebyshev` | 0.410% | 0.367% | 0.812% | 0.006872 |
| european | `floater_hormann` | 0.411% | 0.365% | 0.814% | 0.006861 |
| european | `makima` | 0.411% | 0.365% | 0.814% | 0.006861 |
| european | `pchip` | 0.411% | 0.365% | 0.814% | 0.006861 |
| european | `akima` | 0.411% | 0.365% | 0.814% | 0.006861 |
| european | `natural_cubic_interpolation` | 0.411% | 0.365% | 0.814% | 0.006863 |
| european | `cubic_smoothing_spline` | 0.411% | 0.365% | 0.814% | 0.006863 |
| european | `matern_gp` | 0.411% | 0.365% | 0.814% | 0.006845 |
| european | `loess` | 0.411% | 0.365% | 0.814% | 0.006803 |
| european | `free_knot_spline` | 0.412% | 0.366% | 0.816% | 0.007094 |
| european | `bspline_regression` | 0.413% | 0.366% | 0.816% | 0.006861 |
| european | `bernstein` | 0.413% | 0.366% | 0.818% | 0.006876 |
| european | `linear` | 0.413% | 0.375% | 0.814% | 0.007327 |
| european | `chebyshev` | 0.413% | 0.367% | 0.815% | 0.006847 |
| european | `adaptive_pspline` | 0.413% | 0.365% | 0.814% | 0.006860 |
| european | `pspline` | 0.413% | 0.365% | 0.814% | 0.006860 |
| european | `natural_cubic_regression` | 0.414% | 0.365% | 0.808% | 0.007142 |
