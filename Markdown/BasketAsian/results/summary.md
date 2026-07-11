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
| `moment_lognormal` | 28.450% | 5.947% | 0.023189 |
| `relative_residual_sparse_chebyshev_pca` | 13.200% | 2.496% | 0.018653 |
| `log_factor_sparse_chebyshev_pca` | 8.995% | 1.996% | 0.012734 |
| `pchip_calibrated_log_factor_pca` | 5.873% | 1.820% | 0.009023 |
| `residual_sparse_chebyshev_pca` | 19.085% | 3.339% | 0.002380 |
| `blend_correction_sparse_chebyshev_pca` | 11.452% | 2.295% | 0.006251 |
