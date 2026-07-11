"""Expanded, product-balanced comparison of generic one-dimensional smoothers."""

import argparse
import csv
import sys
from dataclasses import replace
from math import exp, log, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import AmericanMain as american
import AsianMain as asian
import BarrierOptExperiment.BarrierMain as barrier
import EuroMain as euro
from OneDimensionalFitExperiment.spline_methods import METHODS, fit_curve


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "expanded"
CACHE = RESULTS / "datasets"
DETAIL_CSV = RESULTS / "expanded_method_results.csv"
SUMMARY = (
    ROOT.parent
    / "Markdown"
    / "MethodStudy"
    / "results"
    / "expanded"
    / "summary.md"
)
RELATIVE_FLOOR = 0.01
LOG_EPS = 1e-10
TRAIN_BUDGET = 10_000_000
BENCHMARK_PATHS = 500_000


def chebyshev_grid(low, high, points, logarithmic=False):
    nodes = np.cos(np.pi * np.arange(points) / (points - 1))
    if logarithmic:
        low, high = log(low), log(high)
        return np.sort(np.exp(low + 0.5 * (nodes + 1.0) * (high - low)))
    return np.sort(low + 0.5 * (nodes + 1.0) * (high - low))


def score(prediction, benchmark):
    error = np.asarray(prediction) - np.asarray(benchmark)
    absolute = np.abs(error)
    relative = absolute / np.maximum(np.abs(benchmark), RELATIVE_FLOOR)
    meaningful = np.abs(benchmark) >= 0.05
    return {
        "max_rel": float(np.max(relative)),
        "p99_rel": float(np.quantile(relative, 0.99)),
        "p95_rel": float(np.quantile(relative, 0.95)),
        "mean_rel": float(np.mean(relative)),
        "meaningful_max_rel": float(np.max(relative[meaningful]))
        if np.any(meaningful)
        else float(np.max(relative)),
        "mae": float(np.mean(absolute)),
        "max_abs": float(np.max(absolute)),
    }


def save_case(case_id, **arrays):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{case_id}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_or_build(case_id, builder):
    path = CACHE / f"{case_id}.npz"
    if not path.exists():
        save_case(case_id, **builder())
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def common_shifted_european(spots, tau, params, paths, seed):
    rng = np.random.default_rng(seed)
    half = (paths + 1) // 2
    base = rng.standard_normal(half)
    output = np.empty(len(spots))
    for index, spot in enumerate(spots):
        shift = float(euro.payoff_boundary_shift(np.array([spot]), tau, params)[0])
        normals = np.r_[shift + base, shift - base][:paths]
        likelihood = np.exp(-shift * normals + 0.5 * shift * shift)
        growth = np.exp(
            (params.rate - params.div_yield - 0.5 * params.vol**2) * tau
            + params.vol * sqrt(tau) * normals
        )
        output[index] = exp(-params.rate * tau) * np.mean(
            euro.payoff(spot * growth, params) * likelihood
        )
    return output


def european_cases(quick):
    configurations = [
        euro.Params(strike=100, vol=0.10, maturity=0.25, rate=0.01, div_yield=0.00),
        euro.Params(strike=100, vol=0.20, maturity=1.00, rate=0.05, div_yield=0.02),
        euro.Params(strike=100, vol=0.40, maturity=2.00, rate=0.03, div_yield=0.00),
        euro.Params(strike=80, vol=0.25, maturity=1.50, rate=0.07, div_yield=0.01),
        euro.Params(strike=120, vol=0.30, maturity=0.50, rate=0.00, div_yield=0.04),
    ]
    configurations += [replace(params, option_type="put") for params in configurations]
    if quick:
        configurations = configurations[:2]
    fractions = (0.15, 0.50, 0.85) if not quick else (0.50,)
    states = 121 if not quick else 31
    budget = TRAIN_BUDGET if not quick else 50_000
    paths = int(np.ceil(budget / states))
    for config_index, params in enumerate(configurations):
        for fraction in fractions:
            tau = params.maturity * fraction
            case_id = f"european_{config_index:02d}_tau_{fraction:.2f}"

            def build(params=params, tau=tau, config_index=config_index):
                train_spot = euro.delta_space_spot_grid(tau, params, states)
                validation_spot = euro.delta_space_spot_grid(tau, params, 401)
                train_value = common_shifted_european(
                    train_spot, tau, params, paths, 10_000 + 31 * config_index
                )
                return {
                    "x_train": euro.d1_from_spot(train_spot, tau, params),
                    "target": np.log(np.maximum(train_value, 0.0) + LOG_EPS),
                    "x_valid": euro.d1_from_spot(validation_spot, tau, params),
                    "benchmark": euro.black_scholes_value(validation_spot, tau, params),
                }

            yield case_id, "european", load_or_build(case_id, build)


def american_surfaces(params, steps, spot_grid, quick):
    old_steps = american.TEST_STEPS
    old_time = american.BENCHMARK_TIME_STEPS
    old_spot = american.BENCHMARK_SPOT_STEPS
    american.TEST_STEPS = sorted(set(steps))
    american.BENCHMARK_TIME_STEPS = 800 if quick else 2400
    american.BENCHMARK_SPOT_STEPS = 600 if quick else 1400
    try:
        return american.fd_benchmarks(spot_grid, params)
    finally:
        american.TEST_STEPS = old_steps
        american.BENCHMARK_TIME_STEPS = old_time
        american.BENCHMARK_SPOT_STEPS = old_spot


def american_cases(quick):
    configurations = [
        american.Params(rate=0.05, div_yield=0.02, vol=0.20, maturity=1.0),
        american.Params(rate=0.01, div_yield=0.00, vol=0.10, maturity=0.5),
        american.Params(rate=0.08, div_yield=0.00, vol=0.30, maturity=2.0),
        american.Params(rate=0.03, div_yield=0.05, vol=0.40, maturity=1.0),
    ]
    if quick:
        configurations = configurations[:1]
    test_steps = (0, 35, 70) if not quick else (35,)
    states = 121 if not quick else 31
    budget = TRAIN_BUDGET if not quick else 50_000
    paths = int(np.ceil(budget / states))
    for config_index, params in enumerate(configurations):
        dense_spot = np.linspace(0.20 * params.strike, 4.0 * params.strike, 1601)
        wanted = list(test_steps) + [step + 1 for step in test_steps]
        surfaces = american_surfaces(params, wanted, dense_spot, quick)
        dt = params.maturity / american.EXERCISE_STEPS
        drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
        vol_step = params.vol * sqrt(dt)
        for step in test_steps:
            case_id = f"american_{config_index:02d}_step_{step:03d}"

            def build(
                params=params,
                step=step,
                surfaces=surfaces,
                config_index=config_index,
            ):
                train_spot = chebyshev_grid(
                    0.40 * params.strike,
                    2.20 * params.strike,
                    states,
                    logarithmic=True,
                )
                validation_spot = np.exp(
                    np.linspace(
                        log(0.40 * params.strike),
                        log(2.20 * params.strike),
                        401,
                    )
                )
                rng = np.random.default_rng(20_000 + 41 * config_index + step)
                half = (paths + 1) // 2
                base = rng.standard_normal(half)
                continuation = np.empty(states)
                for index, spot in enumerate(train_spot):
                    next_spot = spot * np.exp(
                        drift + vol_step * np.r_[base, -base][:paths]
                    )
                    next_value = np.interp(
                        next_spot,
                        dense_spot,
                        surfaces[step + 1],
                        left=params.strike,
                        right=0.0,
                    )
                    continuation[index] = exp(-params.rate * dt) * np.mean(next_value)
                return {
                    "x_train": np.log(train_spot / params.strike),
                    "target": continuation,
                    "x_valid": np.log(validation_spot / params.strike),
                    "benchmark": np.interp(
                        validation_spot, dense_spot, surfaces[step]
                    ),
                    "intrinsic": american.intrinsic(validation_spot, params),
                }

            yield case_id, "american", load_or_build(case_id, build)


def canonical_asian_states(x, day_index, params):
    x = np.asarray(x)
    m = asian.future_count(day_index, params)
    tau = max(m * asian.daily_dt(params), asian.daily_dt(params))
    q = np.exp(
        (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
        - x * params.vol * sqrt(tau)
    )
    maximum = params.n_fixings * params.strike / (1.0 + m * q)
    if day_index == 0:
        spot = maximum
        running = np.zeros_like(spot)
    else:
        spot = np.minimum(params.strike, 0.90 * maximum)
        running = params.n_fixings * params.strike - spot - m * q * spot
    return spot, np.maximum(running, 0.0)


def common_asian_labels(spots, running, day_index, params, paths, seed):
    values = np.empty(len(spots))
    stderr = np.empty(len(spots))
    for index, (spot, accrued) in enumerate(zip(spots, running)):
        values[index], stderr[index] = asian.simulate_state_value(
            float(spot),
            float(accrued),
            day_index,
            params,
            np.random.default_rng(seed),
            paths,
        )
    return values, stderr


def asian_cases(quick):
    configurations = [
        asian.Params(vol=0.20, maturity=1.0, rate=0.05, div_yield=0.02),
        asian.Params(vol=0.10, maturity=0.5, rate=0.01, div_yield=0.00),
        asian.Params(vol=0.40, maturity=2.0, rate=0.03, div_yield=0.04),
    ]
    if quick:
        configurations = configurations[:1]
    day_indices = (0, 6, 9) if not quick else (6,)
    states = 121 if not quick else 31
    validation_points = 41 if not quick else 11
    budget = TRAIN_BUDGET if not quick else 30_000
    benchmark_paths = BENCHMARK_PATHS if not quick else 2_000
    train_paths = int(np.ceil(budget / states))
    for config_index, params in enumerate(configurations):
        for day_index in day_indices:
            case_id = f"asian_{config_index:02d}_day_{day_index:02d}"

            def build(
                params=params,
                day_index=day_index,
                config_index=config_index,
            ):
                train_x = np.linspace(-5.5, 5.5, states)
                validation_x = np.linspace(-5.35, 5.35, validation_points)
                train_spot, train_running = canonical_asian_states(
                    train_x, day_index, params
                )
                valid_spot, valid_running = canonical_asian_states(
                    validation_x, day_index, params
                )
                train_value, _ = common_asian_labels(
                    train_spot,
                    train_running,
                    day_index,
                    params,
                    train_paths,
                    30_000 + 43 * config_index + day_index,
                )
                benchmark, benchmark_stderr = common_asian_labels(
                    valid_spot,
                    valid_running,
                    day_index,
                    params,
                    benchmark_paths,
                    40_000 + 47 * config_index + day_index,
                )
                train_base = np.maximum(
                    asian.linear_values(
                        train_spot, train_running, day_index, params
                    ),
                    0.0,
                )
                valid_base = np.maximum(
                    asian.linear_values(
                        valid_spot, valid_running, day_index, params
                    ),
                    0.0,
                )
                return {
                    "x_train": train_x,
                    "target": np.log(
                        np.maximum(train_value / train_spot, 0.0) + asian.LOG_EPS
                    ),
                    "target_time": np.log(
                        np.maximum(
                            (train_value - train_base) / train_spot, 0.0
                        )
                        + asian.LOG_EPS
                    ),
                    "x_valid": validation_x,
                    "benchmark": benchmark,
                    "benchmark_stderr": benchmark_stderr,
                    "spot_valid": valid_spot,
                    "base_valid": valid_base,
                }

            yield case_id, "asian", load_or_build(case_id, build)


def barrier_cases(quick):
    configurations = [
        barrier.Params(vol=0.20, maturity=1.0, lower_barrier=75, upper_barrier=135),
        barrier.Params(vol=0.20, maturity=1.0, lower_barrier=85, upper_barrier=120),
        barrier.Params(vol=0.40, maturity=1.0, lower_barrier=60, upper_barrier=155),
        barrier.Params(vol=0.25, maturity=0.5, lower_barrier=80, upper_barrier=130),
    ]
    if quick:
        configurations = configurations[:1]
    months = (0, 9) if not quick else (0,)
    kinds = barrier.BARRIER_KINDS if not quick else ("down_out",)
    states = 121 if not quick else 31
    validation_states = 31 if not quick else 11
    budget = TRAIN_BUDGET if not quick else 30_000
    benchmark_paths = BENCHMARK_PATHS if not quick else 2_000
    train_paths = int(np.ceil(budget / states))
    for config_index, params in enumerate(configurations):
        for month in months:
            for kind in kinds:
                shared_id = (
                    f"barrier_shifted_{config_index:02d}_month_{month:02d}_{kind}"
                )

                def build(
                    params=params,
                    month=month,
                    kind=kind,
                    config_index=config_index,
                ):
                    train_spot = barrier.make_spot_grid(kind, params, states)
                    valid_spot = barrier.make_shifted_spot_grid(
                        kind, params, validation_states
                    )
                    train, _ = barrier.build_labels(
                        train_spot,
                        month,
                        kind,
                        params,
                        np.random.default_rng(
                            50_000 + 53 * config_index + month
                        ),
                        train_paths,
                    )
                    benchmark, benchmark_stderr = barrier.build_labels(
                        valid_spot,
                        month,
                        kind,
                        params,
                        np.random.default_rng(
                            60_000 + 59 * config_index + month
                        ),
                        benchmark_paths,
                    )
                    output = {
                        "x_train": np.log(train_spot),
                        "x_valid": np.log(valid_spot),
                    }
                    for monitoring in barrier.MONITORING_TYPES:
                        output[f"target_{monitoring}"] = np.log(
                            np.maximum(train[monitoring], 0.0) + LOG_EPS
                        )
                        output[f"benchmark_{monitoring}"] = benchmark[monitoring]
                        output[f"stderr_{monitoring}"] = benchmark_stderr[monitoring]
                    return output

                data = load_or_build(shared_id, build)
                for monitoring in barrier.MONITORING_TYPES:
                    yield (
                        f"{shared_id}_{monitoring}",
                        "barrier",
                        {
                            "x_train": data["x_train"],
                            "target": data[f"target_{monitoring}"],
                            "x_valid": data["x_valid"],
                            "benchmark": data[f"benchmark_{monitoring}"],
                            "benchmark_stderr": data[f"stderr_{monitoring}"],
                        },
                    )


def predict_case(data, method):
    curve = fit_curve(data["x_train"], data["target"], method)
    fitted = curve(data["x_valid"])
    if "intrinsic" in data:
        return np.maximum(data["intrinsic"], fitted)
    if "target_time" in data:
        time_curve = fit_curve(data["x_train"], data["target_time"], method)
        spot = data["spot_valid"]
        base = data["base_valid"]
        value_prediction = spot * (
            np.exp(np.clip(fitted, -30.0, 20.0)) - asian.LOG_EPS
        )
        time_prediction = base + spot * (
            np.exp(np.clip(time_curve(data["x_valid"]), -30.0, 20.0))
            - asian.LOG_EPS
        )
        return np.maximum(
            np.where(
                base / np.maximum(spot, 1e-12) > asian.HYBRID_INTRINSIC_SWITCH,
                time_prediction,
                value_prediction,
            ),
            0.0,
        )
    return np.maximum(np.exp(np.clip(fitted, -30.0, 20.0)) - LOG_EPS, 0.0)


def transformed_overshoot(data, method):
    rates = []
    sizes = []
    for target_name in ("target", "target_time"):
        if target_name not in data:
            continue
        x = data["x_train"]
        target = data[target_name]
        fitted = fit_curve(x, target, method)(data["x_valid"])
        index = np.clip(np.searchsorted(x, data["x_valid"]) - 1, 0, len(x) - 2)
        local_low = np.minimum(target[index], target[index + 1])
        local_high = np.maximum(target[index], target[index + 1])
        excursion = np.maximum.reduce(
            (local_low - fitted, fitted - local_high, np.zeros_like(fitted))
        )
        scale = max(float(np.max(target) - np.min(target)), 1e-12)
        rates.append(float(np.mean(excursion > 1e-9 * scale)))
        sizes.append(float(np.max(excursion) / scale))
    return float(np.mean(rates)), float(np.max(sizes))


def write_results(rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with DETAIL_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    products = sorted(set(row["product"] for row in rows))
    product_rows = []
    for method in METHODS:
        for product in products:
            selected = [
                row
                for row in rows
                if row["method"] == method and row["product"] == product
            ]
            product_rows.append(
                {
                    "method": method,
                    "product": product,
                    "average_p99": float(np.mean([row["p99_rel"] for row in selected])),
                    "average_max": float(np.mean([row["max_rel"] for row in selected])),
                    "worst_max": float(np.max([row["max_rel"] for row in selected])),
                    "average_meaningful_max": float(
                        np.mean([row["meaningful_max_rel"] for row in selected])
                    ),
                    "average_mae": float(np.mean([row["mae"] for row in selected])),
                    "overshoot_rate": float(
                        np.mean([row["overshoot_rate"] for row in selected])
                    ),
                }
            )
    overall = []
    for method in METHODS:
        selected = [row for row in product_rows if row["method"] == method]
        overall.append(
            {
                "method": method,
                "balanced_p99": float(np.mean([row["average_p99"] for row in selected])),
                "balanced_average_max": float(
                    np.mean([row["average_max"] for row in selected])
                ),
                "balanced_meaningful_max": float(
                    np.mean([row["average_meaningful_max"] for row in selected])
                ),
                "balanced_overshoot_rate": float(
                    np.mean([row["overshoot_rate"] for row in selected])
                ),
                "worst_case": float(
                    np.max(
                        [
                            row["max_rel"]
                            for row in rows
                            if row["method"] == method
                        ]
                    )
                ),
            }
        )
    overall.sort(key=lambda row: (row["balanced_p99"], row["balanced_average_max"]))

    lines = [
        "# Expanded one-dimensional smoother comparison",
        "",
        "Methods are ranked by the mean product-family p99 relative error. Each",
        "product family receives equal weight. Hyperparameters use training labels only.",
        "",
        "| Rank | Method | Balanced avg p99 | Avg max >= $0.05 | Local overshoot | Raw worst |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(overall, 1):
        lines.append(
            f"| {rank} | `{row['method']}` | {100 * row['balanced_p99']:.3f}% | "
            f"{100 * row['balanced_meaningful_max']:.3f}% | "
            f"{100 * row['balanced_overshoot_rate']:.2f}% | "
            f"{100 * row['worst_case']:.3f}% |"
        )
    lines += [
        "",
        "## Product matrix",
        "",
        "| Product | Method | Avg p99 | Avg max >= $0.05 | Worst max | Avg MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for product in products:
        selected = [row for row in product_rows if row["product"] == product]
        selected.sort(key=lambda row: row["average_p99"])
        for row in selected:
            lines.append(
                f"| {product} | `{row['method']}` | "
                f"{100 * row['average_p99']:.3f}% | "
                f"{100 * row['average_meaningful_max']:.3f}% | "
                f"{100 * row['worst_max']:.3f}% | "
                f"{row['average_mae']:.6f} |"
            )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines[: 8 + len(METHODS)]))
    print(f"results: {RESULTS}")


def run(quick=False):
    rows = []
    generators = (
        european_cases(quick),
        american_cases(quick),
        asian_cases(quick),
        barrier_cases(quick),
    )
    for generator in generators:
        for case_id, product, data in generator:
            for method in METHODS:
                prediction = predict_case(data, method)
                overshoot_rate, max_overshoot = transformed_overshoot(data, method)
                rows.append(
                    {
                        "case_id": case_id,
                        "product": product,
                        "method": method,
                        **score(prediction, data["benchmark"]),
                        "overshoot_rate": overshoot_rate,
                        "max_scaled_overshoot": max_overshoot,
                    }
                )
            print(f"finished {case_id}", flush=True)
    write_results(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    global RESULTS, CACHE, DETAIL_CSV, SUMMARY
    if args.quick:
        RESULTS = ROOT / "results" / "quick"
        CACHE = RESULTS / "datasets"
        DETAIL_CSV = RESULTS / "expanded_method_results.csv"
        SUMMARY = (
            ROOT.parent
            / "Markdown"
            / "MethodStudy"
            / "results"
            / "quick"
            / "summary.md"
        )
    run(args.quick)


if __name__ == "__main__":
    main()
