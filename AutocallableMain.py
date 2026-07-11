import csv
import os
from dataclasses import dataclass, replace
from math import exp, log, sqrt

import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.stats import norm, qmc


@dataclass(frozen=True)
class Params:
    strike: float = 100.0
    notional: float = 100.0
    rate: float = 0.045
    div_yield: float = 0.015
    vol: float = 0.24
    maturity: float = 1.0
    n_observations: int = 4
    autocall_barrier: float = 1.00
    coupon_barrier: float = 0.75
    protection_barrier: float = 0.65
    coupon_per_observation: float = 0.025
    seed: int = 211


METHODS = ["pchip", "akima"]
TEST_OBSERVATION_INDICES = [0, 1, 2, 3]
TRAIN_STATES = 91
VALIDATION_STATES = 181
TRAIN_PATHS = 16_384
BENCHMARK_PATHS = 65_536
RELATIVE_ERROR_FLOOR = 0.05
SPOT_RANGE = (45.0, 180.0)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "AutocallableOptExperiment", "results")
METHOD_CSV = os.path.join(OUTPUT_DIR, "autocallable_proxy_method_results.csv")
DETAIL_CSV = os.path.join(OUTPUT_DIR, "autocallable_proxy_validation_details.csv")
SUMMARY_PATH = os.path.join(
    ROOT, "Markdown", "Autocallable", "results", "summary.md"
)

CASE_GRID = [
    ("base", Params()),
    ("high_vol", replace(Params(), vol=0.34, protection_barrier=0.60)),
    ("low_autocall", replace(Params(), autocall_barrier=0.92, coupon_per_observation=0.018)),
    ("high_coupon", replace(Params(), coupon_per_observation=0.04, vol=0.28)),
    ("downside_heavy", replace(Params(), vol=0.38, protection_barrier=0.75)),
]


def power_of_two_at_least(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 1))))


def sobol_normals(n_paths, dimension, seed):
    count = power_of_two_at_least(n_paths)
    engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    uniforms = engine.random_base2(int(np.log2(count)))
    uniforms = np.clip(uniforms, 1e-12, 1.0 - 1e-12)
    return norm.ppf(uniforms)


def observation_dt(params):
    return params.maturity / params.n_observations


def simulate_values(start_spot, observation_index, params, n_paths, seed):
    start_spot = np.asarray(start_spot, dtype=float)
    remaining = params.n_observations - observation_index
    if remaining <= 0:
        redemption = np.where(
            start_spot >= params.coupon_barrier * params.strike,
            params.notional * (1.0 + params.coupon_per_observation * params.n_observations),
            np.where(
                start_spot >= params.protection_barrier * params.strike,
                params.notional,
                params.notional * start_spot / params.strike,
            ),
        )
        return redemption

    normals = sobol_normals(n_paths, remaining, seed)
    n = len(normals)
    spot = np.repeat(start_spot[:, None], n, axis=1)
    active = np.ones_like(spot, dtype=bool)
    value = np.zeros_like(spot)
    dt = observation_dt(params)
    drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
    vol_step = params.vol * sqrt(dt)

    for step in range(remaining):
        obs_number = observation_index + step + 1
        spot *= np.exp(drift + vol_step * normals[:, step][None, :])
        discount = exp(-params.rate * dt * (step + 1))
        final_step = obs_number == params.n_observations

        if not final_step:
            called = active & (spot >= params.autocall_barrier * params.strike)
            value[called] = (
                discount
                * params.notional
                * (1.0 + params.coupon_per_observation * obs_number)
            )
            active[called] = False
        else:
            coupon_redemption = params.notional * (
                1.0 + params.coupon_per_observation * obs_number
            )
            maturity_value = np.where(
                spot >= params.coupon_barrier * params.strike,
                coupon_redemption,
                np.where(
                    spot >= params.protection_barrier * params.strike,
                    params.notional,
                    params.notional * spot / params.strike,
                ),
            )
            value[active] = discount * maturity_value[active]
            active[:, :] = False

    return np.mean(value, axis=1)


def training_spots():
    x = np.cos(np.pi * np.arange(TRAIN_STATES) / (TRAIN_STATES - 1))
    low, high = log(SPOT_RANGE[0]), log(SPOT_RANGE[1])
    return np.sort(np.exp(low + 0.5 * (x + 1.0) * (high - low)))


def validation_spots():
    return np.exp(np.linspace(log(SPOT_RANGE[0]), log(SPOT_RANGE[1]), VALIDATION_STATES))


def fit_curve(method, spots, values):
    x = np.log(spots)
    y = np.log(np.maximum(values, 0.0) + 1e-10)
    order = np.argsort(x)
    if method == "pchip":
        curve = PchipInterpolator(x[order], y[order], extrapolate=True)
    elif method == "akima":
        curve = Akima1DInterpolator(x[order], y[order])
    else:
        raise ValueError(method)

    def predict(new_spot):
        z = np.clip(np.log(np.maximum(new_spot, 1e-12)), x[order][0], x[order][-1])
        fitted = curve(z)
        return np.maximum(np.exp(np.clip(fitted, -40.0, 20.0)) - 1e-10, 0.0)

    return predict


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
    method_rows = []
    detail_rows = []
    train_grid = training_spots()
    test_grid = validation_spots()

    for obs_index in TEST_OBSERVATION_INDICES:
        train_values = simulate_values(
            train_grid,
            obs_index,
            params,
            TRAIN_PATHS,
            params.seed + 17 * obs_index,
        )
        truth = simulate_values(
            test_grid,
            obs_index,
            params,
            BENCHMARK_PATHS,
            params.seed + 1000 + 37 * obs_index,
        )
        for method in METHODS:
            proxy = fit_curve(method, train_grid, train_values)
            prediction = proxy(test_grid)
            metrics = score(prediction, truth)
            row = {
                "case": case_name,
                "method": method,
                "observation_index": obs_index,
                "time": obs_index * observation_dt(params),
                **metrics,
            }
            method_rows.append(row)
            for spot, actual, fitted in zip(test_grid, truth, prediction):
                detail_rows.append(
                    {
                        "case": case_name,
                        "method": method,
                        "observation_index": obs_index,
                        "spot": float(spot),
                        "benchmark": float(actual),
                        "proxy": float(fitted),
                        "error": float(fitted - actual),
                    }
                )
    return method_rows, detail_rows


def summarize(rows):
    summary = {}
    for method in sorted({row["method"] for row in rows}):
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
        "# Autocallable Proxy Experiment",
        "",
        "Single-underlying autocallable note priced conditionally on not having",
        "called before the current observation date.",
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
            "## Test Cases",
            "",
            "| Case | Vol | Autocall barrier | Coupon barrier | Protection barrier | Coupon/obs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, params in CASE_GRID:
        lines.append(
            f"| {name} | {params.vol:.3f} | {params.autocall_barrier:.3f} | "
            f"{params.coupon_barrier:.3f} | {params.protection_barrier:.3f} | "
            f"{params.coupon_per_observation:.3f} |"
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

    summary = summarize(method_rows)
    print("Autocallable proxy experiment")
    print(f"cases: {len(CASE_GRID)} | train paths/state: {TRAIN_PATHS:,}")
    print(f"benchmark paths/state: {BENCHMARK_PATHS:,}")
    print()
    print("method   worst max rel   avg p99   avg MAE")
    for method, stats in summary.items():
        print(
            f"{method:7s}  {100.0 * stats['worst_max_rel']:11.3f}%  "
            f"{100.0 * stats['avg_p99_rel']:8.3f}%  {stats['avg_mae']:8.6f}"
        )
    print()
    print(f"method CSV written to: {METHOD_CSV}")
    print(f"summary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
