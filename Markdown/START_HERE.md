# Start Here For Future Agents

This repository is a Monte Carlo proxy-pricing research prototype. Read this
file first, then use the product-specific markdown summaries below before
changing code or rerunning heavy experiments.

## Current Layout

- Parent-level `*Main.py` files are intended to be standalone user entry points.
- Experiment subfolders contain exploratory scripts and heavier research code.
- Markdown notes are centralized in this `Markdown/` folder.
- Generated CSV files, plots, caches, and temporary files should stay out of the
  public repository unless the user explicitly asks otherwise.
- The main PDF is `output/pdf/proxy_pricing_methodology.pdf`.

## Recommended Read Order

1. `README.md`
2. `European/findings/european_option_proxy_findings.md`
3. `MethodStudy/findings/one_dimensional_fitting_findings.md`
4. `MethodStudy/findings/expanded_spline_study.md`
5. `Asian/findings/asian_option_proxy_findings.md`
6. `American/findings/american_option_proxy_findings.md`
7. `Barrier/findings/barrier_proxy_findings.md`
8. `BasketCliquet/findings/basket_slv_cliquet_findings.md`
9. `MethodStudy/results/current_performance_matrix.md`
10. `MethodStudy/results/correctness_tests.md`
11. `MethodStudy/results/timing_performance_summary.md`
12. `MethodStudy/results/expanded_coverage_summary.md`
13. Product result summaries under `*/results/summary.md`

## Important Current Decisions

- Use SciPy `PchipInterpolator` as the default one-feature proxy unless a new
  experiment gives a clear reason to replace it.
- Use Sobol low-discrepancy paths and antithetic pairing for MC label generation
  and benchmarks whenever practical.
- Use likelihood-ratio or mixture importance sampling for tail-sensitive labels
  when the payoff has material wing exposure.
- For European calls and puts, the default proxy is log-value PCHIP in the
  Black-Scholes `d1` coordinate.
- For single-feature American, Bermudan, Asian, and barrier studies, prefer
  shape-preserving interpolation over high-degree monomials because it avoids
  artificial oscillation near exercise, barrier, and low-value regions.
- Basket cliquet is not solved by a fitted-only proxy in all cases. The current
  hard-case fallback is a cached Sobol/LR safety proxy.
- New option-type folders should follow the existing layout:
  `Markdown/<OptionType>/results/summary.md`.

## What To Preserve

- Keep parent-level main scripts standalone. They should not import code from
  the experiment subfolders.
- Keep markdown centralized here by option type. If a script writes a summary,
  it should write to `Markdown/<OptionType>/...`.
- Keep public commits limited to source, configuration, markdown, and the PDF
  unless the user approves committing generated result files.
- When adding a new instrument, include:
  - a parent-level standalone main script,
  - an experiment subfolder if useful,
  - a centralized markdown summary,
  - a small grid of test cases rather than one hand-picked case.
