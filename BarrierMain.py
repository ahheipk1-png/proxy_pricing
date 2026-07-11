import csv
from dataclasses import dataclass
from math import comb, exp, log, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.stats import norm, qmc

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


@dataclass(frozen=True)
class Params:
    strike: float = 100.0
    rate: float = 0.05
    div_yield: float = 0.02
    vol: float = 0.20
    maturity: float = 1.0
    n_monitoring_dates: int = 12
    lower_barrier: float = 75.0
    upper_barrier: float = 135.0
    seed: int = 101


METHODS = [
    "pchip",
    "akima",
    "chebyshev",
    "bernstein",
]
BARRIER_KINDS = ["down_out", "up_out", "double_out"]
MONITORING_TYPES = ["discrete", "continuous"]
TEST_MONTHS = [0, 3, 6, 9, 12]
TRAIN_STATES = 121
VALIDATION_STATES = 31
TRAIN_SCENARIOS_PER_FIT = 10_000_000
BENCHMARK_PATHS_PER_STATE = 524_288
SIMULATION_BATCH = 131_072
RELATIVE_ERROR_FLOOR = 0.01
LOG_EPS = 1e-10
RIDGE = 1e-8

OUTPUT_DIR = Path(__file__).resolve().parent / "BarrierOptExperiment" / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
METHOD_CSV = OUTPUT_DIR / "barrier_proxy_method_results.csv"
DETAIL_CSV = OUTPUT_DIR / "barrier_proxy_validation_details.csv"
SUMMARY_PATH = (
    Path(__file__).resolve().parent
    / "Markdown"
    / "Barrier"
    / "results"
    / "summary.md"
)


def dt(params):
    return params.maturity / params.n_monitoring_dates


def power_of_two_at_least(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 1))))


def qmc_path_count(target_paths):
    return 2 * power_of_two_at_least((int(target_paths) + 1) // 2)


def sobol_antithetic_batches(target_paths, dimension, seed, max_batch):
    half_total = power_of_two_at_least((int(target_paths) + 1) // 2)
    half_batch = power_of_two_at_least(max(1, int(max_batch) // 2))
    half_batch = min(half_batch, half_total)
    engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    remaining = half_total
    while remaining:
        half = min(half_batch, remaining)
        base = engine.random(half)
        base = np.clip(base, 1e-12, 1.0 - 1e-12)
        base = norm.ppf(base)
        yield np.vstack((base, -base))
        remaining -= half


def remaining_steps(month, params):
    return params.n_monitoring_dates - month


def discount(month, params):
    return exp(-params.rate * remaining_steps(month, params) * dt(params))


def vanilla_payoff(spot, params):
    return np.maximum(np.asarray(spot) - params.strike, 0.0)


def knock_in_from_parity(vanilla_value, knock_out_value):
    """Zero-rebate knock-in value under the same barrier monitoring convention."""
    return np.maximum(
        np.asarray(vanilla_value) - np.asarray(knock_out_value), 0.0
    )


def barrier_domain(kind, params):
    if kind == "down_out":
        return params.lower_barrier, 180.0
    if kind == "up_out":
        return 45.0, params.upper_barrier
    if kind == "double_out":
        return params.lower_barrier, params.upper_barrier
    raise ValueError(kind)


def is_alive(spot, kind, params):
    spot = np.asarray(spot)
    if kind == "down_out":
        return spot > params.lower_barrier
    if kind == "up_out":
        return spot < params.upper_barrier
    if kind == "double_out":
        return (spot > params.lower_barrier) & (spot < params.upper_barrier)
    raise ValueError(kind)


def make_spot_grid(kind, params, n_points):
    low, high = barrier_domain(kind, params)
    buffer = 1e-4
    log_low = log(low * (1.0 + buffer)) if kind != "up_out" else log(low)
    log_high = log(high * (1.0 - buffer)) if kind != "down_out" else log(high)
    nodes = np.cos(np.pi * np.arange(n_points) / (n_points - 1))
    grid = np.exp(log_low + 0.5 * (nodes + 1.0) * (log_high - log_low))
    return np.sort(grid)


def make_shifted_spot_grid(kind, params, n_points):
    low, high = barrier_domain(kind, params)
    buffer = 1e-4
    log_low = log(low * (1.0 + buffer)) if kind != "up_out" else log(low)
    log_high = log(high * (1.0 - buffer)) if kind != "down_out" else log(high)
    nodes = np.cos(np.pi * (np.arange(n_points) + 0.5) / n_points)
    grid = np.exp(log_low + 0.5 * (nodes + 1.0) * (log_high - log_low))
    return np.sort(grid)


def single_bridge_survival(x, y, barrier, variance, lower):
    if lower:
        valid = (x > barrier) & (y > barrier)
        exponent = -2.0 * (x - barrier) * (y - barrier) / variance
    else:
        valid = (x < barrier) & (y < barrier)
        exponent = -2.0 * (barrier - x) * (barrier - y) / variance
    result = np.zeros_like(y)
    result[valid] = 1.0 - np.exp(np.minimum(exponent[valid], 0.0))
    return np.clip(result, 0.0, 1.0)


def double_bridge_survival(x, y, lower, upper, variance, terms=6):
    valid = (x > lower) & (x < upper) & (y > lower) & (y < upper)
    result = np.zeros_like(y)
    if not np.any(valid):
        return result
    xv = x[valid]
    yv = y[valid]
    width = upper - lower
    direct = yv - xv
    series = np.zeros_like(xv)
    for index in range(-terms, terms + 1):
        image_shift = 2.0 * index * width
        first = direct + image_shift
        second = yv + xv - 2.0 * lower + image_shift
        series += np.exp(
            -np.clip((first * first - direct * direct) / (2.0 * variance), -50.0, 50.0)
        )
        series -= np.exp(
            -np.clip((second * second - direct * direct) / (2.0 * variance), -50.0, 50.0)
        )
    result[valid] = np.clip(series, 0.0, 1.0)
    return result


def simulate_state(spot0, month, kind, params, rng, n_paths):
    if not bool(is_alive(np.array([spot0]), kind, params)[0]):
        return {"discrete": 0.0, "continuous": 0.0}, {
            "discrete": 0.0,
            "continuous": 0.0,
        }
    steps = remaining_steps(month, params)
    if steps == 0:
        value = float(vanilla_payoff(spot0, params))
        return {"discrete": value, "continuous": value}, {
            "discrete": 0.0,
            "continuous": 0.0,
        }

    step_dt = dt(params)
    variance = params.vol**2 * step_dt
    drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * step_dt
    diffusion = params.vol * sqrt(step_dt)
    log_lower = log(params.lower_barrier)
    log_upper = log(params.upper_barrier)
    totals = {name: 0.0 for name in MONITORING_TYPES}
    totals_sq = {name: 0.0 for name in MONITORING_TYPES}
    count = 0
    seed = rng.integers(0, np.iinfo(np.int64).max)
    for normals in sobol_antithetic_batches(n_paths, steps, seed, SIMULATION_BATCH):
        batch = len(normals)
        increments = drift + diffusion * normals
        log_path = log(spot0) + np.cumsum(increments, axis=1)
        terminal = np.exp(log_path[:, -1])
        discounted_payoff = discount(month, params) * vanilla_payoff(terminal, params)

        if kind == "down_out":
            discrete_alive = np.all(log_path > log_lower, axis=1)
        elif kind == "up_out":
            discrete_alive = np.all(log_path < log_upper, axis=1)
        else:
            discrete_alive = np.all(
                (log_path > log_lower) & (log_path < log_upper), axis=1
            )

        continuous_survival = np.ones(batch)
        segment_start = np.full(batch, log(spot0))
        for step in range(steps):
            segment_end = log_path[:, step]
            if kind == "down_out":
                segment_survival = single_bridge_survival(
                    segment_start, segment_end, log_lower, variance, lower=True
                )
            elif kind == "up_out":
                segment_survival = single_bridge_survival(
                    segment_start, segment_end, log_upper, variance, lower=False
                )
            else:
                segment_survival = double_bridge_survival(
                    segment_start,
                    segment_end,
                    log_lower,
                    log_upper,
                    variance,
                )
            continuous_survival *= segment_survival
            segment_start = segment_end

        samples = {
            "discrete": discounted_payoff * discrete_alive,
            "continuous": discounted_payoff * continuous_survival,
        }
        for monitoring, sample in samples.items():
            totals[monitoring] += float(np.sum(sample))
            totals_sq[monitoring] += float(np.sum(sample * sample))
        count += batch

    values = {}
    stderrs = {}
    for monitoring in MONITORING_TYPES:
        mean = totals[monitoring] / count
        sample_variance = max(
            (totals_sq[monitoring] - count * mean * mean) / max(count - 1, 1),
            0.0,
        )
        values[monitoring] = mean
        stderrs[monitoring] = sqrt(sample_variance / count)
    return values, stderrs


def build_labels(spots, month, kind, params, rng, paths):
    # Common random numbers make Monte Carlo label noise smooth across the state grid.
    common_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    values = {
        monitoring: np.empty(len(spots)) for monitoring in MONITORING_TYPES
    }
    stderrs = {
        monitoring: np.empty(len(spots)) for monitoring in MONITORING_TYPES
    }
    for index, spot in enumerate(spots):
        state_rng = np.random.default_rng(common_seed)
        state_values, state_stderrs = simulate_state(
            float(spot), month, kind, params, state_rng, paths
        )
        for monitoring in MONITORING_TYPES:
            values[monitoring][index] = state_values[monitoring]
            stderrs[monitoring][index] = state_stderrs[monitoring]
    return values, stderrs


def bernstein_design(unit_x, degree):
    return np.column_stack(
        [
            comb(degree, index)
            * unit_x**index
            * (1.0 - unit_x) ** (degree - index)
            for index in range(degree + 1)
        ]
    )


def fit_proxy(spots, values, kind, params, method):
    x = np.log(spots)
    target = np.log(np.maximum(values, 0.0) + LOG_EPS)
    if method in {"pchip", "akima"}:
        curve = (
            PchipInterpolator(x, target, extrapolate=True)
            if method == "pchip"
            else Akima1DInterpolator(x, target)
        )
    else:
        low, high = float(x[0]), float(x[-1])
        if method == "chebyshev":
            degree = min(15, len(x) - 1)
            scaled = 2.0 * (x - low) / (high - low) - 1.0
            design = chebvander(scaled, degree)

            def new_design(new_x):
                scaled_new = np.clip(
                    2.0 * (new_x - low) / (high - low) - 1.0, -1.0, 1.0
                )
                return chebvander(scaled_new, degree)

        elif method == "bernstein":
            degree = min(15, len(x) - 1)
            design = bernstein_design((x - low) / (high - low), degree)

            def new_design(new_x):
                unit = np.clip((new_x - low) / (high - low), 0.0, 1.0)
                return bernstein_design(unit, degree)

        else:
            raise ValueError(method)
        penalty = np.eye(design.shape[1]) * RIDGE
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ target
        )
        curve = lambda new_x: new_design(new_x) @ coefficients

    def predict(new_spots):
        new_spots = np.asarray(new_spots)
        new_x = np.clip(np.log(np.maximum(new_spots, 1e-12)), x[0], x[-1])
        fitted = np.maximum(
            np.exp(np.clip(curve(new_x), -30.0, 20.0))
            - LOG_EPS,
            0.0,
        )
        fitted[~is_alive(new_spots, kind, params)] = 0.0
        return fitted

    return predict


def score(prediction, benchmark, stderr):
    error = prediction - benchmark
    absolute = np.abs(error)
    relative = absolute / np.maximum(np.abs(benchmark), RELATIVE_ERROR_FLOOR)
    return {
        "max_rel": float(np.max(relative)),
        "p99_rel": float(np.quantile(relative, 0.99)),
        "p95_rel": float(np.quantile(relative, 0.95)),
        "mae": float(np.mean(absolute)),
        "max_abs": float(np.max(absolute)),
        "median_noise_ratio": float(
            np.median(absolute / np.maximum(stderr, 1e-12))
        ),
    }


def draw_plot(path, title, spots, benchmark, prediction):
    if Image is None:
        return
    width, height = 1200, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panels = [(60, 65, 1140, 350), (60, 390, 1140, 650), (60, 690, 1140, 940)]
    error = prediction - benchmark
    relative = 100.0 * error / np.maximum(np.abs(benchmark), RELATIVE_ERROR_FLOOR)
    series = [
        [("benchmark", benchmark, (35, 90, 180)), ("proxy", prediction, (210, 55, 55))],
        [("signed error", error, (25, 145, 85))],
        [("signed relative error (%)", relative, (125, 70, 170))],
    ]
    draw.text((60, 20), title, fill="black")
    for panel, lines in zip(panels, series):
        left, top, right, bottom = panel
        draw.rectangle(panel, outline=(180, 180, 180))
        all_y = np.concatenate([line[1] for line in lines])
        y_low, y_high = float(np.min(all_y)), float(np.max(all_y))
        margin = max((y_high - y_low) * 0.05, 1e-8)
        y_low, y_high = y_low - margin, y_high + margin
        if y_low < 0.0 < y_high:
            y0 = bottom - (0.0 - y_low) / (y_high - y_low) * (bottom - top)
            draw.line((left, y0, right, y0), fill=(210, 210, 210))
        for line_index, (name, values, color) in enumerate(lines):
            points = []
            for spot, value in zip(spots, values):
                px = left + (spot - spots[0]) / (spots[-1] - spots[0]) * (right - left)
                py = bottom - (value - y_low) / (y_high - y_low) * (bottom - top)
                points.append((px, py))
            draw.line(points, fill=color, width=3)
            draw.text((left + 10 + 230 * line_index, top + 8), name, fill=color)
    image.save(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    params = Params()
    rng = np.random.default_rng(params.seed)
    train_paths = qmc_path_count(np.ceil(TRAIN_SCENARIOS_PER_FIT / TRAIN_STATES))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    method_rows = []
    detail_rows = []
    plot_payload = {}

    for kind in BARRIER_KINDS:
        for month in TEST_MONTHS:
            train_spots = make_spot_grid(kind, params, TRAIN_STATES)
            validation_spots = make_shifted_spot_grid(
                kind, params, VALIDATION_STATES
            )
            train_values, train_stderr = build_labels(
                train_spots, month, kind, params, rng, train_paths
            )
            benchmark, benchmark_stderr = build_labels(
                validation_spots,
                month,
                kind,
                params,
                rng,
                BENCHMARK_PATHS_PER_STATE,
            )
            for monitoring in MONITORING_TYPES:
                variant = f"{kind}_{monitoring}"
                for method in METHODS:
                    proxy = fit_proxy(
                        train_spots,
                        train_values[monitoring],
                        kind,
                        params,
                        method,
                    )
                    prediction = proxy(validation_spots)
                    metrics = score(
                        prediction,
                        benchmark[monitoring],
                        benchmark_stderr[monitoring],
                    )
                    method_rows.append(
                        {
                            "variant": variant,
                            "barrier_kind": kind,
                            "monitoring": monitoring,
                            "method": method,
                            "month": month,
                            "remaining_steps": remaining_steps(month, params),
                            "train_states": TRAIN_STATES,
                            "train_paths_per_state": train_paths,
                            "train_scenarios_used": TRAIN_STATES * train_paths,
                            "validation_states": VALIDATION_STATES,
                            "benchmark_paths_per_state": BENCHMARK_PATHS_PER_STATE,
                            **metrics,
                            "avg_train_stderr": float(
                                np.mean(train_stderr[monitoring])
                            ),
                            "avg_benchmark_stderr": float(
                                np.mean(benchmark_stderr[monitoring])
                            ),
                        }
                    )
                    for index in range(VALIDATION_STATES):
                        error = prediction[index] - benchmark[monitoring][index]
                        detail_rows.append(
                            {
                                "variant": variant,
                                "method": method,
                                "month": month,
                                "spot": validation_spots[index],
                                "benchmark": benchmark[monitoring][index],
                                "proxy": prediction[index],
                                "signed_error": error,
                                "relative_error": abs(error)
                                / max(
                                    abs(benchmark[monitoring][index]),
                                    RELATIVE_ERROR_FLOOR,
                                ),
                                "benchmark_stderr": benchmark_stderr[monitoring][index],
                            }
                        )
                    plot_payload[(variant, month, method)] = (
                        validation_spots,
                        benchmark[monitoring],
                        prediction,
                    )
            print(f"finished {kind}, month {month}")

    write_csv(METHOD_CSV, method_rows)
    write_csv(DETAIL_CSV, detail_rows)
    lines = [
        "# Barrier option proxy experiment",
        "",
        "| Variant | PCHIP worst max | Best method | Best worst max |",
        "|---|---:|---|---:|",
    ]
    for kind in BARRIER_KINDS:
        for monitoring in MONITORING_TYPES:
            variant = f"{kind}_{monitoring}"
            aggregate = {}
            for method in METHODS:
                rows = [
                    row
                    for row in method_rows
                    if row["variant"] == variant
                    and row["method"] == method
                    and row["remaining_steps"] > 0
                ]
                aggregate[method] = max(row["max_rel"] for row in rows)
            best = min(METHODS, key=aggregate.get)
            lines.append(
                f"| `{variant}` | {100 * aggregate['pchip']:.3f}% | "
                f"`{best}` | {100 * aggregate[best]:.3f}% |"
            )
            for month in TEST_MONTHS:
                draw_plot(
                    PLOT_DIR / f"{variant}_month_{month:02d}_pchip.png",
                    f"{variant}, reset month {month}",
                    *plot_payload[(variant, month, "pchip")],
                )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="ascii")
    print()
    print("\n".join(lines))
    print(f"results: {OUTPUT_DIR}")


def main():
    run()


if __name__ == "__main__":
    main()
