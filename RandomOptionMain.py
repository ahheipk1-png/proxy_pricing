import csv
import os
from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.stats import norm, qmc


@dataclass(frozen=True)
class Market:
    rate: float = 0.045
    div_yield: float = 0.015
    vol: float = 0.26
    maturity: float = 1.0
    seed: int = 419


@dataclass(frozen=True)
class PayoffCase:
    name: str
    strikes: np.ndarray
    values: np.ndarray


METHODS = ["pchip", "akima", "chebyshev"]
TIME_FRACTIONS = [0.0, 0.25, 0.50, 0.75]
TRAIN_STATES = 101
VALIDATION_STATES = 251
TRAIN_PATHS = 16_384
BENCHMARK_PATHS = 65_536
CHEBYSHEV_DEGREE = 13
RIDGE = 1e-7
RELATIVE_ERROR_FLOOR = 0.05
SPOT_RANGE = (35.0, 220.0)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "RandomOptionOptExperiment", "results")
METHOD_CSV = os.path.join(OUTPUT_DIR, "random_option_proxy_method_results.csv")
DETAIL_CSV = os.path.join(OUTPUT_DIR, "random_option_proxy_validation_details.csv")
SUMMARY_PATH = os.path.join(ROOT, "Markdown", "RandomOption", "results", "summary.md")


def make_payoff_cases(seed=1234, n_cases=8):
    rng = np.random.default_rng(seed)
    cases = []
    base_strikes = np.linspace(45.0, 190.0, 9)
    for idx in range(n_cases):
        increments = rng.normal(0.0, 12.0, len(base_strikes))
        values = np.cumsum(increments) + rng.uniform(10.0, 35.0)
        values = np.clip(values, 0.0, 90.0)
        if idx % 3 == 0:
            values += np.maximum(base_strikes - 105.0, 0.0) * rng.uniform(0.10, 0.35)
        if idx % 3 == 1:
            values += np.maximum(105.0 - base_strikes, 0.0) * rng.uniform(0.10, 0.35)
        values = np.clip(values, 0.0, 120.0)
        cases.append(PayoffCase(f"random_piecewise_{idx + 1}", base_strikes, values))
    return cases


PAYOFF_CASES = make_payoff_cases()
MARKET_CASES = [
    ("base_market", Market()),
    ("low_vol", Market(vol=0.16, seed=421)),
    ("high_vol", Market(vol=0.38, seed=423)),
]


def payoff(spot, payoff_case):
    spot = np.asarray(spot, dtype=float)
    return np.interp(
        spot,
        payoff_case.strikes,
        payoff_case.values,
        left=float(payoff_case.values[0]),
        right=float(payoff_case.values[-1]),
    )


def power_of_two_at_least(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 1))))


def sobol_normals(n_points, dimension, seed):
    count = power_of_two_at_least(n_points)
    engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    uniforms = engine.random_base2(int(np.log2(count)))
    uniforms = np.clip(uniforms, 1e-12, 1.0 - 1e-12)
    return norm.ppf(uniforms)[:, 0]


def spot_grid(n_points):
    return np.exp(np.linspace(log(SPOT_RANGE[0]), log(SPOT_RANGE[1]), n_points))


def mc_value(start_spot, tau, market, payoff_case, n_paths, seed):
    start_spot = np.asarray(start_spot, dtype=float)
    if tau <= 0.0:
        return payoff(start_spot, payoff_case)
    z = sobol_normals(n_paths, 1, seed)
    terminal = start_spot[:, None] * np.exp(
        (market.rate - market.div_yield - 0.5 * market.vol**2) * tau
        + market.vol * sqrt(tau) * z[None, :]
    )
    return exp(-market.rate * tau) * np.mean(payoff(terminal, payoff_case), axis=1)


def fit_proxy(method, spots, values):
    x = np.log(spots)
    order = np.argsort(x)
    x = x[order]
    y = np.log(np.maximum(values[order], 0.0) + 1e-10)
    x_min, x_max = float(x[0]), float(x[-1])

    if method == "pchip":
        curve = PchipInterpolator(x, y, extrapolate=True)

        def predict(new_spot):
            z = np.clip(np.log(np.maximum(new_spot, 1e-12)), x_min, x_max)
            return np.maximum(np.exp(np.clip(curve(z), -40.0, 20.0)) - 1e-10, 0.0)

        return predict

    if method == "akima":
        curve = Akima1DInterpolator(x, y)

        def predict(new_spot):
            z = np.clip(np.log(np.maximum(new_spot, 1e-12)), x_min, x_max)
            return np.maximum(np.exp(np.clip(curve(z), -40.0, 20.0)) - 1e-10, 0.0)

        return predict

    if method == "chebyshev":
        scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        design = chebvander(scaled, CHEBYSHEV_DEGREE)
        lhs = design.T @ design + RIDGE * np.eye(design.shape[1])
        coef = np.linalg.solve(lhs, design.T @ y)

        def predict(new_spot):
            z = np.clip(np.log(np.maximum(new_spot, 1e-12)), x_min, x_max)
            scaled_new = 2.0 * (z - x_min) / (x_max - x_min) - 1.0
            fitted = chebvander(scaled_new, CHEBYSHEV_DEGREE) @ coef
            return np.maximum(np.exp(np.clip(fitted, -40.0, 20.0)) - 1e-10, 0.0)

        return predict

    raise ValueError(method)


def score(prediction, truth):
    error = prediction - truth
    abs_error = np.abs(error)
    rel_error = abs_error / np.maximum(np.abs(truth), RELATIVE_ERROR_FLOOR)
    return {
        "max_rel": float(np.max(rel_error)),
        "p99_rel": float(np.quantile(rel_error, 0.99)),
        "p95_rel": float(np.quantile(rel_error, 0.95)),
        "mae": float(np.mean(abs_error)),
        "max_abs": float(np.max(abs_error)),
    }


def run_case(market_name, market, payoff_case):
    train_spots = spot_grid(TRAIN_STATES)
    test_spots = spot_grid(VALIDATION_STATES)
    method_rows = []
    detail_rows = []

    for time_fraction in TIME_FRACTIONS:
        tau = market.maturity * (1.0 - time_fraction)
        train_values = mc_value(
            train_spots,
            tau,
            market,
            payoff_case,
            TRAIN_PATHS,
            market.seed + int(1000 * time_fraction),
        )
        truth = mc_value(
            test_spots,
            tau,
            market,
            payoff_case,
            BENCHMARK_PATHS,
            market.seed + 10_000 + int(1000 * time_fraction),
        )
        for method in METHODS:
            proxy = fit_proxy(method, train_spots, train_values)
            prediction = proxy(test_spots)
            metrics = score(prediction, truth)
            row = {
                "market_case": market_name,
                "payoff_case": payoff_case.name,
                "method": method,
                "time_fraction": time_fraction,
                **metrics,
            }
            method_rows.append(row)
            for spot, actual, fitted in zip(test_spots, truth, prediction):
                detail_rows.append(
                    {
                        "market_case": market_name,
                        "payoff_case": payoff_case.name,
                        "method": method,
                        "time_fraction": time_fraction,
                        "spot": float(spot),
                        "benchmark": float(actual),
                        "proxy": float(fitted),
                        "error": float(fitted - actual),
                    }
                )
    return method_rows, detail_rows


def summarize(rows):
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            "worst_max_rel": max(row["max_rel"] for row in subset),
            "avg_p99_rel": float(np.mean([row["p99_rel"] for row in subset])),
            "avg_mae": float(np.mean([row["mae"] for row in subset])),
        }
    return summary


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows):
    summary = summarize(rows)
    best = min(
        summary,
        key=lambda method: (
            summary[method]["worst_max_rel"],
            summary[method]["avg_p99_rel"],
            summary[method]["avg_mae"],
        ),
    )
    lines = [
        "# Random Option Proxy Experiment",
        "",
        "Fixed-seed random piecewise-linear terminal payoffs under GBM. This is a",
        "generic one-feature stress test for the interpolation proxy rather than a",
        "named exchange-traded product.",
        "",
        f"Best method on this grid: `{best}`.",
        "",
        "| Method | Worst max relative error | Average p99 | Average MAE |",
        "|---|---:|---:|---:|",
    ]
    for method, stats in summary.items():
        lines.append(
            f"| {method} | {100.0 * stats['worst_max_rel']:.3f}% | "
            f"{100.0 * stats['avg_p99_rel']:.3f}% | {stats['avg_mae']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Test Grid",
            "",
            f"- Payoff cases: {len(PAYOFF_CASES)} random piecewise-linear terminal payoffs.",
            f"- Market cases: {len(MARKET_CASES)} volatility regimes.",
            f"- Time fractions: {', '.join(str(x) for x in TIME_FRACTIONS)}.",
        ]
    )
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    method_rows = []
    detail_rows = []
    for market_name, market in MARKET_CASES:
        for payoff_case in PAYOFF_CASES:
            rows, details = run_case(market_name, market, payoff_case)
            method_rows.extend(rows)
            detail_rows.extend(details)

    write_csv(METHOD_CSV, method_rows)
    write_csv(DETAIL_CSV, detail_rows)
    write_summary(method_rows)
    summary = summarize(method_rows)

    print("Random option proxy experiment")
    print(f"payoff cases: {len(PAYOFF_CASES)} | market cases: {len(MARKET_CASES)}")
    print(f"train paths/state: {TRAIN_PATHS:,} | benchmark paths/state: {BENCHMARK_PATHS:,}")
    print()
    print("method     worst max rel   avg p99   avg MAE")
    for method, stats in summary.items():
        print(
            f"{method:9s}  {100.0 * stats['worst_max_rel']:11.3f}%  "
            f"{100.0 * stats['avg_p99_rel']:8.3f}%  {stats['avg_mae']:8.6f}"
        )
    print()
    print(f"method CSV written to: {METHOD_CSV}")
    print(f"summary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
