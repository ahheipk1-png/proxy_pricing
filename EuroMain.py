import csv
from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
from numpy.polynomial.chebyshev import chebvander


@dataclass(frozen=True)
class Params:
    s0: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    div_yield: float = 0.02
    vol: float = 0.20
    maturity: float = 1.0
    seed: int = 7
    option_type: str = "call"


METHOD_NAME = "log Chebyshev"
METHOD_DETAIL = "d1, degree=7"
STATE_POINTS = 121
MC_PATHS_PER_STATE = 25_000
GRID_POINTS = 501
TIME_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
RELATIVE_ERROR_FLOOR = 0.01
SHIFT_BUFFER = 0.5
SHIFT_CAP = 4.0
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "EuroOptExperiment"
    / "default_run"
    / "euro_default_results.csv"
)


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def payoff(spot, params):
    spot = np.asarray(spot, dtype=float)
    if params.option_type == "call":
        return np.maximum(spot - params.strike, 0.0)
    if params.option_type == "put":
        return np.maximum(params.strike - spot, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_value(spot, tau, params):
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return payoff(spot, params)

    sigma_sqrt_tau = params.vol * sqrt(tau)
    d1 = (
        np.log(spot / params.strike)
        + (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    ) / sigma_sqrt_tau
    d2 = d1 - sigma_sqrt_tau
    discounted_spot = spot * exp(-params.div_yield * tau)
    discounted_strike = params.strike * exp(-params.rate * tau)

    if params.option_type == "call":
        return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    if params.option_type == "put":
        return discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def d1_from_spot(spot, tau, params):
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return (spot - params.strike) / params.strike
    return (
        np.log(spot / params.strike)
        + (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    ) / (params.vol * sqrt(tau))


def spot_from_d1(d1, tau, params):
    d1 = np.asarray(d1, dtype=float)
    if tau <= 0.0:
        return params.strike * (1.0 + 0.01 * d1)
    return params.strike * np.exp(
        d1 * params.vol * sqrt(tau)
        - (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    )


def delta_space_spot_grid(tau, params, n_points, low_delta=1e-4, high_delta=1.0 - 1e-4):
    if tau <= 0.0:
        return np.linspace(50.0, 190.0, n_points)
    normal = NormalDist()
    d1_grid = np.linspace(normal.inv_cdf(low_delta), normal.inv_cdf(high_delta), n_points)
    return spot_from_d1(d1_grid, tau, params)


def payoff_boundary_shift(spot, tau, params):
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return np.zeros_like(spot)

    threshold = (
        np.log(params.strike / spot)
        - (params.rate - params.div_yield - 0.5 * params.vol**2) * tau
    ) / (params.vol * sqrt(tau))

    if params.option_type == "call":
        return np.clip(threshold + SHIFT_BUFFER, 0.0, SHIFT_CAP)
    if params.option_type == "put":
        return np.clip(threshold - SHIFT_BUFFER, -SHIFT_CAP, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def shifted_mc_value(spot, tau, params, rng):
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return payoff(spot, params)

    half_paths = (MC_PATHS_PER_STATE + 1) // 2
    base = rng.standard_normal((spot.size, half_paths))
    shift = payoff_boundary_shift(spot, tau, params)
    shifted_normals = np.concatenate(
        (shift[:, None] + base, shift[:, None] - base), axis=1
    )[:, :MC_PATHS_PER_STATE]

    likelihood_ratio = np.exp(
        -shift[:, None] * shifted_normals + 0.5 * shift[:, None] ** 2
    )
    growth = np.exp(
        (params.rate - params.div_yield - 0.5 * params.vol**2) * tau
        + params.vol * sqrt(tau) * shifted_normals
    )
    terminal_spot = spot[:, None] * growth
    return exp(-params.rate * tau) * np.mean(
        payoff(terminal_spot, params) * likelihood_ratio, axis=1
    )


def weighted_ridge(design, target, ridge=1e-8):
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def fit_log_chebyshev_d1(spot, values, tau, params, degree=7):
    x = d1_from_spot(spot, tau, params)
    x_min = float(x.min())
    x_max = float(x.max())

    def scale(new_x):
        return np.clip(2.0 * (new_x - x_min) / (x_max - x_min) - 1.0, -1.0, 1.0)

    target = np.log(np.maximum(values, 0.0) + 1e-10)
    coeffs = weighted_ridge(chebvander(scale(x), degree), target)

    def predict(new_spot):
        new_x = d1_from_spot(new_spot, tau, params)
        return np.exp(chebvander(scale(new_x), degree) @ coeffs) - 1e-10

    return predict


def score(prediction, truth):
    error = prediction - truth
    abs_error = np.abs(error)
    rel_error = abs_error / np.maximum(np.abs(truth), RELATIVE_ERROR_FLOOR)
    return {
        "max_rel": float(rel_error.max()),
        "p99_rel": float(np.quantile(rel_error, 0.99)),
        "mae": float(abs_error.mean()),
        "max_abs": float(abs_error.max()),
    }


def run_time_slice(params, time_fraction, rng):
    time = params.maturity * time_fraction
    tau = params.maturity - time
    grid_spot = delta_space_spot_grid(tau, params, GRID_POINTS)
    truth = black_scholes_value(grid_spot, tau, params)

    if tau <= 0.0:
        prediction = payoff(grid_spot, params)
    else:
        state_spot = delta_space_spot_grid(tau, params, STATE_POINTS)
        state_value = shifted_mc_value(state_spot, tau, params, rng)
        proxy = fit_log_chebyshev_d1(state_spot, state_value, tau, params)
        prediction = proxy(grid_spot)

    return {
        "time": time,
        "tau": tau,
        "method": METHOD_NAME,
        "detail": METHOD_DETAIL,
        **score(prediction, truth),
    }


def write_results(rows):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    params = Params()
    rng = np.random.default_rng(params.seed + 197)
    rows = [run_time_slice(params, fraction, rng) for fraction in TIME_FRACTIONS]
    write_results(rows)

    print("European default proxy test")
    print(f"method: {METHOD_NAME} | {METHOD_DETAIL}")
    print(f"training states/time: {STATE_POINTS}")
    print(f"shifted MC paths/state: {MC_PATHS_PER_STATE:,}")
    print()
    print("time    max rel    p99 rel    MAE       max abs")
    print("----    -------    -------    ---       -------")
    for row in rows:
        print(
            f"{row['time']:4.2f}    {100.0 * row['max_rel']:7.3f}%   "
            f"{100.0 * row['p99_rel']:7.3f}%   "
            f"{row['mae']:8.6f}  {row['max_abs']:8.6f}"
        )
    print()
    print(f"results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
