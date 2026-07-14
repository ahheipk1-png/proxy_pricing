# Basket Exotic Option Experiment

`BasketExoticMain.py` runs the basket generic exotic proxy study.

The script prices at least 100 four-underlying payoff configurations using the
same data-driven pipeline as the single-underlying study, with weighted-basket
and order-statistic rankings.

Generated CSV files and plots are written to `BasketExoticOptExperiment/results/`
and are intentionally ignored by Git. The committed summary is written to
`Markdown/BasketExotic/results/summary.md`.
