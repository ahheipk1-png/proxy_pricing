# Literature-inspired SLV basket cliquet surrogates

| Variant | Method | Worst max | Average p99 | Average MAE |
|---|---|---:|---:|---:|
| `basket_return` | `sparse_chebyshev_2001` | 10.258% | 3.646% | 0.017434 |
| `basket_return` | `residual_ensemble_5001` | 71.543% | 18.326% | 0.037859 |
| `basket_return` | `weighted_sparse_chebyshev_5001` | 78.834% | 17.348% | 0.016705 |
| `basket_return` | `weighted_sparse_chebyshev_2001` | 78.918% | 17.018% | 0.020424 |
| `basket_return` | `direct_hermite_5001` | 87.225% | 20.976% | 0.068821 |
| `basket_return` | `residual_hermite_5001` | 87.523% | 20.788% | 0.067821 |
| `basket_return` | `residual_nystrom_5001` | 91.143% | 20.673% | 0.043607 |
| `basket_return` | `direct_hermite_2001` | 94.530% | 23.999% | 0.091069 |
| `basket_return` | `residual_hermite_2001` | 94.641% | 23.821% | 0.089543 |
| `basket_return` | `residual_local_5001` | 108.617% | 22.217% | 0.026776 |
| `basket_return` | `normal_moment_baseline` | 133.213% | 31.633% | 0.132524 |
| `basket_return` | `direct_ensemble_5001` | 671.502% | 126.258% | 0.223083 |
| `basket_return` | `direct_nystrom_5001` | 730.520% | 136.323% | 0.178762 |
| `basket_return` | `direct_local_5001` | 4105.523% | 780.852% | 0.709577 |
| `average_clipped` | `sparse_chebyshev_2001` | 11.566% | 3.824% | 0.015884 |
| `average_clipped` | `weighted_sparse_chebyshev_5001` | 20.520% | 5.987% | 0.010643 |
| `average_clipped` | `weighted_sparse_chebyshev_2001` | 25.185% | 4.964% | 0.012448 |
| `average_clipped` | `residual_local_5001` | 29.111% | 10.814% | 0.034357 |
| `average_clipped` | `residual_nystrom_5001` | 41.475% | 12.061% | 0.034216 |
| `average_clipped` | `normal_moment_baseline` | 42.761% | 20.540% | 0.282596 |
| `average_clipped` | `residual_ensemble_5001` | 45.412% | 11.299% | 0.029654 |
| `average_clipped` | `residual_hermite_5001` | 60.799% | 15.040% | 0.044652 |
| `average_clipped` | `direct_hermite_5001` | 61.025% | 14.820% | 0.049984 |
| `average_clipped` | `residual_hermite_2001` | 74.830% | 18.666% | 0.049973 |
| `average_clipped` | `direct_hermite_2001` | 75.670% | 18.787% | 0.056917 |
| `average_clipped` | `direct_ensemble_5001` | 369.122% | 155.274% | 0.216540 |
| `average_clipped` | `direct_nystrom_5001` | 503.823% | 196.504% | 0.215501 |
| `average_clipped` | `direct_local_5001` | 2493.853% | 962.739% | 0.700032 |
| `worst_of` | `sparse_chebyshev_2001` | 9.862% | 6.138% | 0.010614 |
| `worst_of` | `weighted_sparse_chebyshev_5001` | 16.885% | 7.711% | 0.008715 |
| `worst_of` | `weighted_sparse_chebyshev_2001` | 18.042% | 7.864% | 0.007743 |
| `worst_of` | `direct_hermite_5001` | 53.457% | 14.553% | 0.016613 |
| `worst_of` | `direct_hermite_2001` | 61.953% | 16.068% | 0.015356 |
| `worst_of` | `residual_local_5001` | 70.175% | 31.929% | 0.053569 |
| `worst_of` | `residual_ensemble_5001` | 73.674% | 28.204% | 0.033494 |
| `worst_of` | `residual_hermite_5001` | 75.654% | 22.339% | 0.023403 |
| `worst_of` | `residual_nystrom_5001` | 76.135% | 32.794% | 0.040106 |
| `worst_of` | `residual_hermite_2001` | 82.528% | 23.731% | 0.025072 |
| `worst_of` | `normal_moment_baseline` | 85.587% | 71.599% | 0.096242 |
| `worst_of` | `direct_ensemble_5001` | 149.714% | 85.253% | 0.172007 |
| `worst_of` | `direct_nystrom_5001` | 304.672% | 102.157% | 0.104691 |
| `worst_of` | `direct_local_5001` | 2042.260% | 560.378% | 0.527324 |
