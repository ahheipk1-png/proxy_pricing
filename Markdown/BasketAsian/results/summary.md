# 10-asset basket Asian proxy experiment

Monthly arithmetic Asian call on an equal-weight basket of 10 correlated GBM underlyings.
The correlation matrix intentionally includes both positive and negative correlations.

Default method: `pchip_calibrated_log_factor_pca`.

Training labels use Sobol low-discrepancy paths, boundary-enriched state sampling,
and a true two-component Gaussian likelihood-ratio importance sampler when it reduces variance.
Current IS policy: use LR mixture when future fixings >= 10; plain Sobol otherwise.
Training state-scenarios per date are about 33,619,968.
Benchmark paths per validation state are 524,288.

| Method | Worst Max % Error | Avg P99 % Error | Avg MAE |
|---|---:|---:|---:|
| `moment_lognormal` | 28.823% | 7.135% | 0.008893 |
| `relative_residual_sparse_chebyshev_pca` | 14.906% | 3.304% | 0.022765 |
| `log_factor_sparse_chebyshev_pca` | 8.378% | 2.196% | 0.015269 |
| `pchip_calibrated_log_factor_pca` | 5.831% | 1.886% | 0.009369 |
| `residual_sparse_chebyshev_pca` | 16.080% | 3.350% | 0.002366 |
| `blend_correction_sparse_chebyshev_pca` | 9.877% | 2.617% | 0.007350 |
