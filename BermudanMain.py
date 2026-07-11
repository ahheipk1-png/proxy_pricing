import csv
import os
from dataclasses import dataclass, replace
from math import exp, log, sqrt

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm, qmc


@dataclass(frozen=True)
class Params:
    strike: float = 100.0
    rate: float = 0.045
    div_yield: float = 0.015
    vol: float = 0.24
    maturity: float = 1.0
    n_exercise_dates: int = 12
    seed: int = 307


TEST_EXERCISE_INDICES = [0, 3, 6, 9, 12]
TRAIN_STATES = 181
PATHS_PER_STATE = 8192
VALIDATION_STATES = 301
BINOMIAL_STEPS_PER_PERIOD = 24
TRAIN_SPOT_RANGE = (20.0, 420.0)
VALIDATION_SPOT_RANGE = (35.0, 230.0)
RELATIVE_ERROR_FLOOR = 0.05
IMPORTANCE_SHIFT_BUFFER = 0.5
IMPORTANCE_SHIFT_CAP = 2.5

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "BermudanOptExperiment", "results")
METHOD_CSV = os.path.join(OUTPUT_DIR, "bermudan_proxy_method_results.csv")
DETAIL_CSV = os.path.join(OUTPUT_DIR, "bermudan_proxy_validation_details.csv")
SUMMARY_PATH = os.path.join(ROOT, "Markdown", "Bermudan", "results", "summary.md")

CASE_GRID = [
    ("base_put", Params()),
    ("low_vol", replace(Params(), vol=0.16)),
    ("high_vol", replace(Params(), vol=0.36)),
    ("deep_itm", replace(Params(), strike=115.0)),
    ("dividend_rich", replace(Params(), div_yield=0.06, vol=0.28)),
]


def intrinsic(spot, params):
    return np.maximum(params.strike - np.asarray(spot, dtype=float), 0.0)


def power_of_two_at_least(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 1))))


def sobol_normals(n_points, dimension, seed):
    count = power_of_two_at_least(n_points)
    engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    uniforms = engine.random_base2(int(np.log2(count)))
    uniforms = np.clip(uniforms, 1e-12, 1.0 - 1e-12)
    return norm.ppf(uniforms)


def training_spots():
    nodes = np.cos(np.pi * np.arange(TRAIN_STATES) / (TRAIN_STATES - 1))
    low, high = log(TRAIN_SPOT_RANGE[0]), log(TRAIN_SPOT_RANGE[1])
    return np.sort(np.exp(low + 0.5 * (nodes + 1.0) * (high - low)))


def validation_spots():
    return np.exp(
        np.linspace(log(VALIDATION_SPOT_RANGE[0]), log(VALIDATION_SPOT_RANGE[1]), VALIDATION_STATES)
    )


def train_proxy(params):
    dt = params.maturity / params.n_exercise_dates
    df = exp(-params.rate * dt)
    drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
    vol_step = params.vol * sqrt(dt)
    normals = sobol_normals(PATHS_PER_STATE, params.n_exercise_dates, params.seed).T
    spots = training_spots()
    value = lambda new_spot: intrinsic(new_spot, params)
    saved = {params.n_exercise_dates: value}

    for step in range(params.n_exercise_dates - 1, -1, -1):
        z = normals[step]
        threshold = (np.log(params.strike / spots) - drift) / vol_step
        shift = np.clip(
            threshold - IMPORTANCE_SHIFT_BUFFER,
            -IMPORTANCE_SHIFT_CAP,
            0.0,
        )
        half = max(len(z) // 2, 1)
        base = z[:half]
        actual = np.concatenate(
            (base[None, :] + 0.0 * shift[:, None], base[None, :] + shift[:, None]),
            axis=1,
        )
        shifted_over_base = np.exp(
            np.clip(actual * shift[:, None] - 0.5 * shift[:, None] ** 2, -50.0, 50.0)
        )
        likelihood_ratio = 1.0 / (0.5 + 0.5 * shifted_over_base)
        next_spot = spots[:, None] * np.exp(drift + vol_step * actual)
        continuation = df * np.mean(value(next_spot) * likelihood_ratio, axis=1)
        log_spots = np.log(spots)
        log_min, log_max = float(log_spots[0]), float(log_spots[-1])
        target = np.log(np.maximum(continuation, 0.0) + 1e-12)
        curve = PchipInterpolator(log_spots, target, extrapolate=True)

        def current_value(new_spot, curve=curve, log_min=log_min, log_max=log_max):
            safe_spot = np.maximum(new_spot, 1e-12)
            continuation_x = np.clip(np.log(safe_spot), log_min, log_max)
            continuation_value = np.maximum(
                np.exp(np.clip(curve(continuation_x), -45.0, 20.0)) - 1e-12,
                0.0,
            )
            return np.maximum(intrinsic(safe_spot, params), continuation_value)

        value = current_value
        if step in TEST_EXERCISE_INDICES:
            saved[step] = value
    return saved


def binomial_bermudan_value(spot, exercise_index, params):
    remaining_periods = params.n_exercise_dates - exercise_index
    if remaining_periods <= 0:
        return intrinsic(spot, params)

    total_steps = remaining_periods * BINOMIAL_STEPS_PER_PERIOD
    dt = (params.maturity / params.n_exercise_dates) / BINOMIAL_STEPS_PER_PERIOD
    u = exp(params.vol * sqrt(dt))
    d = 1.0 / u
    growth = exp((params.rate - params.div_yield) * dt)
    p = (growth - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    df = exp(-params.rate * dt)

    j = np.arange(total_steps + 1)
    terminal = spot * (u**j) * (d ** (total_steps - j))
    values = intrinsic(terminal, params)

    for step in range(total_steps - 1, -1, -1):
        values = df * (p * values[1:] + (1.0 - p) * values[:-1])
        if step % BINOMIAL_STEPS_PER_PERIOD == 0:
            j = np.arange(step + 1)
            nodes = spot * (u**j) * (d ** (step - j))
            values = np.maximum(values, intrinsic(nodes, params))
    return float(values[0])


def benchmark_values(spots, exercise_index, params):
    return np.array(
        [binomial_bermudan_value(float(spot), exercise_index, params) for spot in spots],
        dtype=float,
    )


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


def run_case(case_name, params):
    proxies = train_proxy(params)
    spots = validation_spots()
    method_rows = []
    detail_rows = []
    for exercise_index in TEST_EXERCISE_INDICES:
        prediction = proxies[exercise_index](spots)
        truth = benchmark_values(spots, exercise_index, params)
        metrics = score(prediction, truth)
        method_rows.append(
            {
                "case": case_name,
                "method": "pchip_dynamic_programming",
                "exercise_index": exercise_index,
                "time": exercise_index * params.maturity / params.n_exercise_dates,
                **metrics,
            }
        )
        for spot, actual, fitted in zip(spots, truth, prediction):
            detail_rows.append(
                {
                    "case": case_name,
                    "method": "pchip_dynamic_programming",
                    "exercise_index": exercise_index,
                    "spot": float(spot),
                    "benchmark": float(actual),
                    "proxy": float(fitted),
                    "error": float(fitted - actual),
                }
            )
    return method_rows, detail_rows


def summarize(rows):
    return {
        "worst_max_rel": max(row["max_rel"] for row in rows),
        "avg_p99_rel": float(np.mean([row["p99_rel"] for row in rows])),
        "avg_mae": float(np.mean([row["mae"] for row in rows])),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows):
    stats = summarize(rows)
    lines = [
        "# Bermudan Option Proxy Experiment",
        "",
        "Monthly-exercise Bermudan put proxy using Sobol MC dynamic programming",
        "and SciPy PCHIP continuation curves.",
        "",
        "| Method | Worst max relative error | Average p99 | Average MAE |",
        "|---|---:|---:|---:|",
        (
            f"| pchip_dynamic_programming | {100.0 * stats['worst_max_rel']:.3f}% | "
            f"{100.0 * stats['avg_p99_rel']:.3f}% | {stats['avg_mae']:.6f} |"
        ),
        "",
        "Benchmark: independent Cox-Ross-Rubinstein Bermudan tree with exercise",
        f"allowed every {BINOMIAL_STEPS_PER_PERIOD} tree steps.",
        "",
        "## Test Cases",
        "",
        "| Case | Strike | Vol | Rate | Dividend yield |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, params in CASE_GRID:
        lines.append(
            f"| {name} | {params.strike:.2f} | {params.vol:.3f} | "
            f"{params.rate:.3f} | {params.div_yield:.3f} |"
        )
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    method_rows = []
    detail_rows = []
    for case_name, params in CASE_GRID:
        rows, details = run_case(case_name, params)
        method_rows.extend(rows)
        detail_rows.extend(details)

    write_csv(METHOD_CSV, method_rows)
    write_csv(DETAIL_CSV, detail_rows)
    write_summary(method_rows)
    stats = summarize(method_rows)

    print("Bermudan put proxy experiment")
    print(f"cases: {len(CASE_GRID)} | paths/state: {PATHS_PER_STATE:,}")
    print(
        f"worst max rel: {100.0 * stats['worst_max_rel']:.3f}% | "
        f"avg p99: {100.0 * stats['avg_p99_rel']:.3f}% | "
        f"avg MAE: {stats['avg_mae']:.6f}"
    )
    print(f"method CSV written to: {METHOD_CSV}")
    print(f"summary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
