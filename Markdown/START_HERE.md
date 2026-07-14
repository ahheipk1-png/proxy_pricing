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
- The main PDF is `output/pdf/proxy_pricing_methodology.pdf`, generated from
  `Documentation/proxy_pricing_methodology.tex` with Tectonic.

## Recommended Read Order

1. `README.md`
2. `European/findings/european_option_proxy_findings.md`
3. `European/findings/european_vol_rate_proxy.md`
4. `European/findings/european_term_structure_proxy.md`
5. `MethodStudy/findings/one_dimensional_fitting_findings.md`
6. `MethodStudy/findings/expanded_spline_study.md`
7. `MethodStudy/findings/feature_engineering_guide.md`
8. `MethodStudy/findings/term_structure_feature_notes.md`
9. `MethodStudy/results/term_structure_single_feature_summary.md`
10. `MethodStudy/findings/method_motivation_and_selection.md`
11. `Asian/findings/asian_option_proxy_findings.md`
12. `American/findings/american_option_proxy_findings.md`
13. `Barrier/findings/barrier_proxy_findings.md`
14. `BasketCliquet/findings/basket_slv_cliquet_findings.md`
15. `SingleExotic/results/summary.md`
16. `BasketExotic/results/summary.md`
17. `Rainbow/results/summary.md`
18. `YieldSeeker/results/summary.md`
19. `Himalayan/results/summary.md`
20. `Lookback/results/summary.md`
21. `Barrier/generic_family_results/summary.md`
22. `Binary/results/summary.md`
23. `MethodStudy/results/current_performance_matrix.md`
24. `MethodStudy/results/correctness_tests.md`
25. `MethodStudy/results/timing_performance_summary.md`
26. `MethodStudy/results/expanded_coverage_summary.md`
27. Product result summaries under `*/results/summary.md`

## Important Current Decisions

- Use SciPy `PchipInterpolator` as the default one-feature proxy unless a new
  experiment gives a clear reason to replace it.
- Use `MethodStudy/findings/feature_engineering_guide.md` before changing
  state variables or adding high-dimensional features; feature sufficiency is
  usually more important than the fitting method.
- Use `MethodStudy/findings/term_structure_feature_notes.md` before adding
  rate or volatility term-structure features. For Europeans, collapse
  deterministic curves to integrated rate and variance; for path-dependent
  products, use event-date summaries before raw curve knots.
- Use `MethodStudy/findings/method_motivation_and_selection.md` before adding
  new fitting methods; it explains why each method exists and when one should
  beat another.
- Use Sobol low-discrepancy paths and antithetic pairing for MC label generation
  and benchmarks whenever practical.
- Use likelihood-ratio or mixture importance sampling for tail-sensitive labels
  when the payoff has material wing exposure.
- For European calls and puts, the default proxy is log-value PCHIP in the
  Black-Scholes `d1` coordinate.
- For single-feature American, Bermudan, Asian, and barrier studies, prefer
  shape-preserving interpolation over high-degree monomials because it avoids
  artificial oscillation near exercise, barrier, and low-value regions.
- The generic exotic pipeline is implemented in `ExoticPipelineCore.py`.
  `SingleExoticMain.py` and `BasketExoticMain.py` run the single-underlying
  and basket versions without replacing the older product-specific scripts.
- The family-level generic exotic scripts are the preferred coverage reports:
  `RainbowMain.py`, `YieldSeekerMain.py`, `HimalayanMain.py`,
  `LookbackMain.py`, `BarrierFamilyMain.py`, and `BinaryMain.py`. Each runs
  100 single-underlying and 100 basket configurations.
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
