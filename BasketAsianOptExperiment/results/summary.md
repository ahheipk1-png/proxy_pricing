# 10-asset basket Asian proxy experiment

Monthly arithmetic Asian call on an equal-weight basket of 10 correlated GBM underlyings.
The correlation matrix intentionally includes both positive and negative correlations.

Default method: `pchip_calibrated_log_factor_pca`.

Training labels use Sobol low-discrepancy paths and boundary-enriched state sampling.
This script does not yet use likelihood-ratio importance sampling for the path simulation measure.
Training state-scenarios per date are about 33,619,968.
Benchmark paths per validation state are 524,288.

| Method | Worst Max % Error | Avg P99 % Error | Avg MAE |
|---|---:|---:|---:|
| `moment_lognormal` | 20.477% | 4.878% | 0.014864 |
| `relative_residual_sparse_chebyshev_pca` | 11.378% | 2.321% | 0.013642 |
| `log_factor_sparse_chebyshev_pca` | 9.187% | 1.955% | 0.009087 |
| `pchip_calibrated_log_factor_pca` | 6.530% | 1.560% | 0.007594 |
| `residual_sparse_chebyshev_pca` | 10.205% | 2.217% | 0.000489 |
| `blend_correction_sparse_chebyshev_pca` | 9.747% | 1.761% | 0.004266 |
