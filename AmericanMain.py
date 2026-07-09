"""Standalone American put proxy using MC dynamic programming and PCHIP."""

import csv
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Params:
    strike: float = 100.0
    rate: float = 0.05
    div_yield: float = 0.02
    vol: float = 0.20
    maturity: float = 1.0
    seed: int = 59


EXERCISE_STEPS = 100
TRAIN_STATES = 121
TOTAL_TRAIN_TRANSITIONS = 10_000_000
PATHS_PER_STATE = int(np.ceil(TOTAL_TRAIN_TRANSITIONS / EXERCISE_STEPS / TRAIN_STATES))
BENCHMARK_TIME_STEPS = 4000
BENCHMARK_SPOT_STEPS = 2000
VALIDATION_POINTS = 401
TEST_STEPS = [0, 20, 40, 60, 80, 100]
SPOT_MIN, SPOT_MAX = 40.0, 220.0
RELATIVE_ERROR_FLOOR = 0.01
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "AmericanOptExperiment"
    / "default_run"
    / "american_default_results.csv"
)


def intrinsic(spot, params):
    return np.maximum(params.strike - np.asarray(spot), 0.0)


def training_spots():
    nodes = np.cos(np.pi * np.arange(TRAIN_STATES) / (TRAIN_STATES - 1))
    low, high = np.log(SPOT_MIN), np.log(SPOT_MAX)
    return np.sort(np.exp(low + 0.5 * (nodes + 1.0) * (high - low)))


def pchip(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    same_sign = delta[:-1] * delta[1:] > 0.0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    denominator = (
        w1 / np.where(delta[:-1] == 0.0, 1.0, delta[:-1])
        + w2 / np.where(delta[1:] == 0.0, 1.0, delta[1:])
    )
    slopes[1:-1] = np.where(same_sign, (w1 + w2) / denominator, 0.0)
    slopes[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (
        h[0] + h[1]
    )
    slopes[-1] = (
        (2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]
    ) / (h[-1] + h[-2])
    for index, secant in ((0, delta[0]), (-1, delta[-1])):
        if slopes[index] * secant <= 0.0:
            slopes[index] = 0.0
        elif abs(slopes[index]) > 3.0 * abs(secant):
            slopes[index] = 3.0 * secant

    def predict(new_x):
        new_x = np.asarray(new_x)
        index = np.clip(np.searchsorted(x, new_x) - 1, 0, len(x) - 2)
        local_h = x[index + 1] - x[index]
        t = np.clip((new_x - x[index]) / local_h, 0.0, 1.0)
        return np.maximum(
            (2 * t**3 - 3 * t**2 + 1) * y[index]
            + (t**3 - 2 * t**2 + t) * local_h * slopes[index]
            + (-2 * t**3 + 3 * t**2) * y[index + 1]
            + (t**3 - t**2) * local_h * slopes[index + 1],
            0.0,
        )

    return predict


def train_proxy(params):
    rng = np.random.default_rng(params.seed)
    half_paths = (PATHS_PER_STATE + 1) // 2
    normals = rng.standard_normal((EXERCISE_STEPS, half_paths))
    dt = params.maturity / EXERCISE_STEPS
    df = exp(-params.rate * dt)
    drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
    vol_step = params.vol * sqrt(dt)
    spots = training_spots()
    value = lambda new_spot: intrinsic(new_spot, params)
    saved = {EXERCISE_STEPS: value}

    for step in range(EXERCISE_STEPS - 1, -1, -1):
        z = normals[step]
        up = spots[:, None] * np.exp(drift + vol_step * z)[None, :]
        down = spots[:, None] * np.exp(drift - vol_step * z)[None, :]
        continuation = 0.5 * df * np.mean(value(up) + value(down), axis=1)
        continuation_proxy = pchip(np.log(spots), continuation)

        def current_value(new_spot, continuation_proxy=continuation_proxy):
            return np.maximum(
                intrinsic(new_spot, params),
                continuation_proxy(np.log(np.maximum(new_spot, 1e-12))),
            )

        value = current_value
        if step in TEST_STEPS:
            saved[step] = value
    return saved


def fd_benchmarks(validation_spot, params):
    s_max = 4.0 * params.strike
    spot_grid = np.linspace(0.0, s_max, BENCHMARK_SPOT_STEPS + 1)
    ds = spot_grid[1]
    dt = params.maturity / BENCHMARK_TIME_STEPS
    values = intrinsic(spot_grid, params)
    exercise = values.copy()
    interior = spot_grid[1:-1]
    diffusion = 0.5 * params.vol**2 * interior**2 / ds**2
    convection = (params.rate - params.div_yield) * interior / (2.0 * ds)
    lower = -dt * (diffusion - convection)
    diagonal = 1.0 + dt * (2.0 * diffusion + params.rate)
    upper = -dt * (diffusion + convection)
    wanted = {
        int(round((1.0 - step / EXERCISE_STEPS) * BENCHMARK_TIME_STEPS)): step
        for step in TEST_STEPS
    }
    output = {100: intrinsic(validation_spot, params)}

    for time_step in range(1, BENCHMARK_TIME_STEPS + 1):
        rhs = values[1:-1].copy()
        rhs[0] -= lower[0] * params.strike
        guess = np.maximum(values[1:-1], exercise[1:-1])
        for _ in range(30):
            old = guess.copy()
            for parity in (0, 1):
                index = np.arange(parity, len(guess), 2)
                left = np.where(index == 0, params.strike, guess[index - 1])
                right = np.where(
                    index == len(guess) - 1,
                    0.0,
                    guess[np.minimum(index + 1, len(guess) - 1)],
                )
                candidate = (
                    rhs[index] - lower[index] * left - upper[index] * right
                ) / diagonal[index]
                guess[index] = np.maximum(
                    exercise[index + 1],
                    guess[index] + 1.25 * (candidate - guess[index]),
                )
            if np.max(np.abs(guess - old)) < 2e-11:
                break
        values[0], values[-1] = params.strike, 0.0
        values[1:-1] = guess
        if time_step in wanted:
            output[wanted[time_step]] = np.interp(
                validation_spot, spot_grid, values
            )
    return output


def main():
    params = Params()
    spot = np.exp(np.linspace(np.log(SPOT_MIN), np.log(SPOT_MAX), VALIDATION_POINTS))
    proxies = train_proxy(params)
    benchmark = fd_benchmarks(spot, params)
    rows = []
    for step in TEST_STEPS:
        prediction = proxies[step](spot)
        error = prediction - benchmark[step]
        absolute = np.abs(error)
        relative = absolute / np.maximum(np.abs(benchmark[step]), RELATIVE_ERROR_FLOOR)
        rows.append(
            {
                "step": step,
                "time": step / EXERCISE_STEPS,
                "max_rel": float(np.max(relative)),
                "p99_rel": float(np.quantile(relative, 0.99)),
                "mae": float(np.mean(absolute)),
                "max_abs": float(np.max(absolute)),
            }
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("American put default proxy test")
    print("method: MC dynamic programming + shape-preserving PCHIP")
    print(f"training transitions: {EXERCISE_STEPS * TRAIN_STATES * PATHS_PER_STATE:,}")
    print()
    print("time    max rel    p99 rel    MAE       max abs")
    for row in rows:
        print(
            f"{row['time']:4.2f}    {100 * row['max_rel']:7.3f}%   "
            f"{100 * row['p99_rel']:7.3f}%   {row['mae']:8.6f}  "
            f"{row['max_abs']:8.6f}"
        )
    print(f"\nresults written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
