# Single Exotic Option Experiment

`SingleExoticMain.py` runs the single-underlying generic exotic proxy study.

The script prices at least 100 data-driven payoff configurations using the
pipeline:

```text
Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff
```

Generated CSV files and plots are written to `SingleExoticOptExperiment/results/`
and are intentionally ignored by Git. The committed summary is written to
`Markdown/SingleExotic/results/summary.md`.
