import csv
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander

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
    seed: int = 53


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
METHOD_CSV = OUTPUT_DIR / "american_proxy_method_results.csv"
DETAIL_CSV = OUTPUT_DIR / "american_proxy_validation_details.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"

EXERCISE_STEPS = 100
TRAIN_STATES = 121
TOTAL_TRAIN_TRANSITIONS = 10_000_000
PATHS_PER_STATE = int(np.ceil(TOTAL_TRAIN_TRANSITIONS / EXERCISE_STEPS / TRAIN_STATES))
BENCHMARK_TIME_STEPS = 4000
BENCHMARK_SPOT_STEPS = 2000
VALIDATION_POINTS = 401
TEST_STEP_INDICES = [0, 20, 40, 60, 80, 100]
SPOT_MIN = 40.0
SPOT_MAX = 220.0
RELATIVE_ERROR_FLOOR = 0.01
RIDGE = 1e-8
METHODS = [
    "direct_chebyshev_d9",
    "linear_spline",
    "pchip_spline",
    "akima_spline",
    "log_chebyshev_d7",
]


def intrinsic(spot, params):
    return np.maximum(params.strike - np.asarray(spot), 0.0)


def scale_log_spot(spot):
    low = np.log(SPOT_MIN)
    high = np.log(SPOT_MAX)
    return np.clip(2.0 * (np.log(np.maximum(spot, 1e-12)) - low) / (high - low) - 1.0, -1.0, 1.0)


def training_spots(params):
    # Chebyshev extrema provide extra resolution in both wings.
    nodes = np.cos(np.pi * np.arange(TRAIN_STATES) / (TRAIN_STATES - 1))
    log_low, log_high = np.log(SPOT_MIN), np.log(SPOT_MAX)
    spots = np.exp(log_low + 0.5 * (nodes + 1.0) * (log_high - log_low))
    return np.sort(spots)


def ridge_fit(design, target):
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def fit_continuation(spot, continuation, method):
    if method in {"linear_spline", "pchip_spline", "akima_spline"}:
        x = np.log(spot)
        y = np.asarray(continuation)
        if method == "linear_spline":
            def predict(new_spot):
                return np.maximum(np.interp(np.log(new_spot), x, y), 0.0)
            return predict

        h = np.diff(x)
        delta = np.diff(y) / h
        if method == "akima_spline" and len(x) >= 5:
            extended = np.empty(len(delta) + 4)
            extended[2:-2] = delta
            extended[1] = 2.0 * delta[0] - delta[1]
            extended[0] = 2.0 * extended[1] - delta[0]
            extended[-2] = 2.0 * delta[-1] - delta[-2]
            extended[-1] = 2.0 * extended[-2] - delta[-1]
            slopes = np.empty_like(y)
            for index in range(len(y)):
                left_far = extended[index]
                left = extended[index + 1]
                right = extended[index + 2]
                right_far = extended[index + 3]
                weight_left = abs(right_far - right)
                weight_right = abs(left - left_far)
                if weight_left + weight_right > 1e-14:
                    slopes[index] = (
                        weight_left * left + weight_right * right
                    ) / (weight_left + weight_right)
                else:
                    slopes[index] = 0.5 * (left + right)
        else:
            slopes = np.zeros_like(y)
            interior = delta[:-1] * delta[1:] > 0.0
            w1 = 2.0 * h[1:] + h[:-1]
            w2 = h[1:] + 2.0 * h[:-1]
            harmonic = (w1 + w2) / (
                w1 / np.where(delta[:-1] == 0.0, 1.0, delta[:-1])
                + w2 / np.where(delta[1:] == 0.0, 1.0, delta[1:])
            )
            slopes[1:-1] = np.where(interior, harmonic, 0.0)
            slopes[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (
                h[0] + h[1]
            )
            slopes[-1] = (
                (2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]
            ) / (h[-1] + h[-2])
            if slopes[0] * delta[0] <= 0.0:
                slopes[0] = 0.0
            elif abs(slopes[0]) > 3.0 * abs(delta[0]):
                slopes[0] = 3.0 * delta[0]
            if slopes[-1] * delta[-1] <= 0.0:
                slopes[-1] = 0.0
            elif abs(slopes[-1]) > 3.0 * abs(delta[-1]):
                slopes[-1] = 3.0 * delta[-1]

        def predict(new_spot):
            new_x = np.log(np.maximum(new_spot, 1e-12))
            index = np.clip(np.searchsorted(x, new_x) - 1, 0, len(x) - 2)
            local_h = x[index + 1] - x[index]
            t = np.clip((new_x - x[index]) / local_h, 0.0, 1.0)
            h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
            h10 = t**3 - 2.0 * t**2 + t
            h01 = -2.0 * t**3 + 3.0 * t**2
            h11 = t**3 - t**2
            fitted = (
                h00 * y[index]
                + h10 * local_h * slopes[index]
                + h01 * y[index + 1]
                + h11 * local_h * slopes[index + 1]
            )
            return np.maximum(fitted, 0.0)
        return predict

    if method == "direct_chebyshev_d9":
        degree = 9
        target = continuation
        inverse = lambda value: np.maximum(value, 0.0)
    else:
        degree = int(method.rsplit("d", 1)[1])
        target = np.log(np.maximum(continuation, 0.0) + 1e-10)
        inverse = lambda value: np.maximum(np.exp(np.clip(value, -30.0, 20.0)) - 1e-10, 0.0)
    coeffs = ridge_fit(chebvander(scale_log_spot(spot), degree), target)

    def predict(new_spot):
        fitted = chebvander(scale_log_spot(new_spot), degree) @ coeffs
        return inverse(fitted)

    return predict


def backward_proxy(params, method, normals):
    dt = params.maturity / EXERCISE_STEPS
    df = exp(-params.rate * dt)
    drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
    vol_step = params.vol * sqrt(dt)
    spots = training_spots(params)
    next_value = lambda new_spot: intrinsic(new_spot, params)
    saved = {EXERCISE_STEPS: next_value}

    for step in range(EXERCISE_STEPS - 1, -1, -1):
        z = normals[step]
        growth_up = np.exp(drift + vol_step * z)
        growth_down = np.exp(drift - vol_step * z)
        next_up = spots[:, None] * growth_up[None, :]
        next_down = spots[:, None] * growth_down[None, :]
        continuation = 0.5 * df * np.mean(
            next_value(next_up) + next_value(next_down), axis=1
        )
        continuation_proxy = fit_continuation(spots, continuation, method)
        previous_value = next_value

        def value(new_spot, continuation_proxy=continuation_proxy):
            return np.maximum(intrinsic(new_spot, params), continuation_proxy(new_spot))

        next_value = value
        if step in TEST_STEP_INDICES:
            saved[step] = next_value
    return saved


def american_put_fd_benchmarks(requested_steps, validation_spot, params):
    # Independent projected implicit finite-difference benchmark.
    s_max = 4.0 * params.strike
    spot_grid = np.linspace(0.0, s_max, BENCHMARK_SPOT_STEPS + 1)
    ds = spot_grid[1]
    dt = params.maturity / BENCHMARK_TIME_STEPS
    values = intrinsic(spot_grid, params)
    exercise = values.copy()
    interior_spot = spot_grid[1:-1]
    diffusion = 0.5 * params.vol**2 * interior_spot**2 / ds**2
    convection = (params.rate - params.div_yield) * interior_spot / (2.0 * ds)
    lower = -dt * (diffusion - convection)
    diagonal = 1.0 + dt * (2.0 * diffusion + params.rate)
    upper = -dt * (diffusion + convection)
    requested_tau_steps = {
        int(round((1.0 - step / EXERCISE_STEPS) * BENCHMARK_TIME_STEPS)): step
        for step in requested_steps
    }
    output = {}
    if 0 in requested_tau_steps:
        output[requested_tau_steps[0]] = intrinsic(validation_spot, params)

    for time_step in range(1, BENCHMARK_TIME_STEPS + 1):
        rhs = values[1:-1].copy()
        rhs[0] -= lower[0] * params.strike
        guess = np.maximum(values[1:-1], exercise[1:-1])
        # Red-black projected SOR converges quickly for this diagonally dominant system.
        for _ in range(30):
            old = guess.copy()
            for parity in (0, 1):
                indices = np.arange(parity, len(guess), 2)
                left = np.where(indices == 0, params.strike, guess[indices - 1])
                right = np.where(
                    indices == len(guess) - 1, 0.0, guess[np.minimum(indices + 1, len(guess) - 1)]
                )
                candidate = (
                    rhs[indices]
                    - lower[indices] * left
                    - upper[indices] * right
                ) / diagonal[indices]
                guess[indices] = np.maximum(
                    exercise[indices + 1],
                    guess[indices] + 1.25 * (candidate - guess[indices]),
                )
            if np.max(np.abs(guess - old)) < 2e-11:
                break
        values[0] = params.strike
        values[-1] = 0.0
        values[1:-1] = guess
        if time_step in requested_tau_steps:
            step = requested_tau_steps[time_step]
            output[step] = np.interp(validation_spot, spot_grid, values)
    return output


def score(prediction, benchmark):
    error = prediction - benchmark
    absolute = np.abs(error)
    relative = absolute / np.maximum(np.abs(benchmark), RELATIVE_ERROR_FLOOR)
    return {
        "max_rel": float(np.max(relative)),
        "p99_rel": float(np.quantile(relative, 0.99)),
        "p95_rel": float(np.quantile(relative, 0.95)),
        "mae": float(np.mean(absolute)),
        "max_abs": float(np.max(absolute)),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_plot(path, step, spot, benchmark, prediction):
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
    draw.text((60, 20), f"American put proxy, time={step / EXERCISE_STEPS:.2f}", fill="black")
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
        for line_idx, (name, y, color) in enumerate(lines):
            points = []
            for xv, yv in zip(spot, y):
                px = left + (xv - spot[0]) / (spot[-1] - spot[0]) * (right - left)
                py = bottom - (yv - y_low) / (y_high - y_low) * (bottom - top)
                points.append((px, py))
            draw.line(points, fill=color, width=3)
            draw.text((left + 10 + 230 * line_idx, top + 8), name, fill=color)
    image.save(path)


def run():
    params = Params()
    rng = np.random.default_rng(params.seed)
    # Antithetic transition samples: half this many normals create PATHS_PER_STATE paths.
    half_paths = (PATHS_PER_STATE + 1) // 2
    normals = rng.standard_normal((EXERCISE_STEPS, half_paths))
    validation_spot = np.exp(
        np.linspace(np.log(SPOT_MIN), np.log(SPOT_MAX), VALIDATION_POINTS)
    )
    benchmarks = american_put_fd_benchmarks(
        TEST_STEP_INDICES, validation_spot, params
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    method_rows = []
    detail_rows = []
    predictions = {}

    for method in METHODS:
        proxies = backward_proxy(params, method, normals)
        for step in TEST_STEP_INDICES:
            tau = params.maturity * (1.0 - step / EXERCISE_STEPS)
            benchmark = benchmarks[step]
            prediction = proxies[step](validation_spot)
            metrics = score(prediction, benchmark)
            method_rows.append(
                {
                    "method": method,
                    "step": step,
                    "time": step / EXERCISE_STEPS,
                    "tau": tau,
                    "exercise_steps": EXERCISE_STEPS,
                    "train_states_per_step": TRAIN_STATES,
                    "paths_per_state": PATHS_PER_STATE,
                    "total_train_transitions": EXERCISE_STEPS
                    * TRAIN_STATES
                    * PATHS_PER_STATE,
                    "benchmark_time_steps": BENCHMARK_TIME_STEPS,
                    "benchmark_spot_steps": BENCHMARK_SPOT_STEPS,
                    **metrics,
                }
            )
            for idx in range(VALIDATION_POINTS):
                error = prediction[idx] - benchmark[idx]
                detail_rows.append(
                    {
                        "method": method,
                        "step": step,
                        "time": step / EXERCISE_STEPS,
                        "spot": validation_spot[idx],
                        "benchmark": benchmark[idx],
                        "proxy": prediction[idx],
                        "signed_error": error,
                        "relative_error": abs(error)
                        / max(abs(benchmark[idx]), RELATIVE_ERROR_FLOOR),
                    }
                )
            predictions[(method, step)] = (benchmark, prediction)
        print(f"finished {method}")

    write_csv(METHOD_CSV, method_rows)
    write_csv(DETAIL_CSV, detail_rows)
    nonterminal = [row for row in method_rows if row["tau"] > 0.0]
    aggregate = {}
    for method in METHODS:
        rows = [row for row in nonterminal if row["method"] == method]
        aggregate[method] = {
            "worst_max": max(row["max_rel"] for row in rows),
            "avg_p99": float(np.mean([row["p99_rel"] for row in rows])),
            "avg_mae": float(np.mean([row["mae"] for row in rows])),
        }
    best = min(METHODS, key=lambda name: aggregate[name]["worst_max"])
    lines = [
        "# American put proxy experiment",
        "",
        f"Best method: `{best}`.",
        "",
        "| Method | Worst max relative error | Average p99 | Average MAE |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        item = aggregate[method]
        lines.append(
            f"| `{method}` | {100 * item['worst_max']:.3f}% | "
            f"{100 * item['avg_p99']:.3f}% | {item['avg_mae']:.6f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="ascii")
    for step in TEST_STEP_INDICES:
        benchmark, prediction = predictions[(best, step)]
        draw_plot(
            PLOT_DIR / f"american_step_{step:03d}_{best}.png",
            step,
            validation_spot,
            benchmark,
            prediction,
        )
    print()
    print(f"best method: {best}")
    print(f"worst max relative error: {100 * aggregate[best]['worst_max']:.3f}%")
    print(f"results: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
