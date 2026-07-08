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
    n_fixings: int = 12
    seed: int = 11
    option_type: str = "call"


METHOD_NAME = "adjusted-moneyness hybrid"
METHOD_DETAIL = "monthly Asian, log value degree=19, log time-value degree=19"
TEST_DAY_INDICES = [0, 3, 6, 9, 11]
ADJUSTED_MONEYNESS_POINTS = 121
VALIDATION_SPOT_POINTS = 9
VALIDATION_AVG_POINTS = 7
TRAIN_SCENARIOS_PER_FIT = 10_000_000
BENCHMARK_PATHS_PER_STATE = 500_000
SIMULATION_BATCH_PATHS = 100_000
HYBRID_VALUE_DEGREE = 19
HYBRID_TIME_VALUE_DEGREE = 19
HYBRID_INTRINSIC_SWITCH = 0.05
RIDGE = 1e-8
LOG_EPS = 1e-12
RELATIVE_ERROR_FLOOR = 0.01
SHIFT_BUFFER = 0.5
SHIFT_CAP = 4.0
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "AsianOptExperiment"
    / "default_run"
    / "asian_default_results.csv"
)


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def payoff_from_average(avg, params):
    avg = np.asarray(avg, dtype=float)
    if params.option_type == "call":
        return np.maximum(avg - params.strike, 0.0)
    if params.option_type == "put":
        return np.maximum(params.strike - avg, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def daily_dt(params):
    return params.maturity / (params.n_fixings - 1)


def future_count(day_index, params):
    return params.n_fixings - day_index - 1


def discount(day_index, params):
    return exp(-params.rate * future_count(day_index, params) * daily_dt(params))


def d1_like_spot(spot, tau, params):
    tau = max(float(tau), daily_dt(params))
    return (
        np.log(np.asarray(spot, dtype=float) / params.strike)
        + (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    ) / (params.vol * sqrt(tau))


def spot_from_d1(d1, tau, params):
    tau = max(float(tau), daily_dt(params))
    return params.strike * np.exp(
        np.asarray(d1, dtype=float) * params.vol * sqrt(tau)
        - (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    )


def adjusted_strike(spot, running_sum_before, day_index, params):
    m = future_count(day_index, params)
    if m == 0:
        return np.full_like(np.asarray(spot, dtype=float), np.nan, dtype=float)
    return (
        params.n_fixings * params.strike
        - np.asarray(running_sum_before, dtype=float)
        - np.asarray(spot, dtype=float)
    ) / m


def expected_future_growth_sum(day_index, params):
    m = future_count(day_index, params)
    dt = daily_dt(params)
    return sum(exp((params.rate - params.div_yield) * j * dt) for j in range(1, m + 1))


def expected_future_sum(spot, day_index, params):
    return np.asarray(spot, dtype=float) * expected_future_growth_sum(day_index, params)


def exact_linear_value(spot, running_sum_before, day_index, params):
    m = future_count(day_index, params)
    total_fixings = params.n_fixings
    current_sum = running_sum_before + spot
    if m == 0:
        return payoff_from_average(current_sum / total_fixings, params)

    expected_average = (
        current_sum + expected_future_sum(spot, day_index, params)
    ) / total_fixings
    return discount(day_index, params) * (expected_average - params.strike)


def linear_values(spots, running_sums, day_index, params):
    return np.array(
        [
            exact_linear_value(float(spot), float(running_sum), day_index, params)
            for spot, running_sum in zip(spots, running_sums)
        ],
        dtype=float,
    )


def adjusted_moneyness_coordinate(spot, running_sum_before, day_index, params):
    m = future_count(day_index, params)
    if m == 0:
        return np.zeros_like(np.asarray(spot, dtype=float))
    strike_adj = np.maximum(
        adjusted_strike(spot, running_sum_before, day_index, params), 1e-12
    )
    tau = max(m * daily_dt(params), daily_dt(params))
    return (
        np.log(np.asarray(spot, dtype=float) / strike_adj)
        + (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    ) / (params.vol * sqrt(tau))


def make_state_grid(day_index, params, spot_points, avg_points):
    tau = future_count(day_index, params) * daily_dt(params)
    normal = NormalDist()
    d1_nodes = np.linspace(normal.inv_cdf(0.001), normal.inv_cdf(0.999), spot_points)
    spot_nodes = spot_from_d1(d1_nodes, max(tau, daily_dt(params)), params)

    if day_index == 0:
        avg_nodes = np.array([params.strike])
    else:
        avg_nodes = params.strike * np.exp(np.linspace(log(0.65), log(1.35), avg_points))

    spots = []
    running_sums = []
    for avg_before in avg_nodes:
        for spot in spot_nodes:
            spots.append(float(spot))
            running_sums.append(float(day_index * avg_before))
    return np.array(spots), np.array(running_sums)


def q_from_adjusted_d1(d1, day_index, params):
    m = future_count(day_index, params)
    tau = max(m * daily_dt(params), daily_dt(params))
    return np.exp(
        (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
        - np.asarray(d1, dtype=float) * params.vol * sqrt(tau)
    )


def make_adjusted_moneyness_grid(day_index, params, n_points):
    if day_index == 0:
        return make_state_grid(day_index, params, 61, 1)

    m = future_count(day_index, params)
    if m == 0:
        return make_state_grid(day_index, params, 15, 11)

    d1_nodes = np.unique(
        np.r_[np.linspace(-6.0, 6.0, n_points), np.linspace(-2.0, 2.0, n_points)]
    )
    q_nodes = q_from_adjusted_d1(d1_nodes, day_index, params)
    spots = []
    running_sums = []
    for q in q_nodes:
        max_valid_spot = params.n_fixings * params.strike / (1.0 + m * q)
        spot = min(params.strike, 0.90 * max_valid_spot)
        running_sum = params.n_fixings * params.strike - spot - m * q * spot
        spots.append(float(spot))
        running_sums.append(float(max(running_sum, 0.0)))
    return np.array(spots), np.array(running_sums)


def geometric_asian_exact(spot, running_sum_before, day_index, params):
    m = future_count(day_index, params)
    if m == 0:
        return 0.0

    total_fixings = params.n_fixings
    strike_adj = (total_fixings * params.strike - running_sum_before - spot) / m
    if params.option_type == "call" and strike_adj <= 0.0:
        return exact_linear_value(spot, running_sum_before, day_index, params)
    if params.option_type == "put" and strike_adj <= 0.0:
        return 0.0

    dt = daily_dt(params)
    drift = params.rate - params.div_yield - 0.5 * params.vol**2
    mean_log = log(spot) + drift * dt * (m + 1) / 2.0
    weights = np.arange(m, 0, -1, dtype=float) / m
    variance_log = params.vol**2 * dt * float(np.sum(weights**2))
    stdev_log = sqrt(max(variance_log, 1e-16))
    d1 = (mean_log - log(strike_adj) + variance_log) / stdev_log
    d2 = d1 - stdev_log
    forward_geo = exp(mean_log + 0.5 * variance_log)
    df = discount(day_index, params)
    scale = m / total_fixings

    if params.option_type == "call":
        geo_value = df * (forward_geo * normal_cdf(d1) - strike_adj * normal_cdf(d2))
    else:
        geo_value = df * (strike_adj * normal_cdf(-d2) - forward_geo * normal_cdf(-d1))
    return float(scale * geo_value)


def bridge_shift_direction(m):
    weights = np.arange(m, 0, -1, dtype=float) / m
    norm = float(np.linalg.norm(weights))
    if norm <= 0.0:
        return weights
    return weights / norm


def shift_amount(spot, running_sum_before, day_index, params):
    m = future_count(day_index, params)
    if m == 0:
        return 0.0
    strike_adj = (params.n_fixings * params.strike - running_sum_before - spot) / m
    if strike_adj <= 0.0:
        return 0.0

    dt = daily_dt(params)
    drift = params.rate - params.div_yield - 0.5 * params.vol**2
    mean_log = log(spot) + drift * dt * (m + 1) / 2.0
    stdev_log = params.vol * sqrt(dt) * float(np.linalg.norm(np.arange(m, 0, -1) / m))
    threshold = (log(strike_adj) - mean_log) / max(stdev_log, 1e-12)

    if params.option_type == "call":
        return float(np.clip(threshold + SHIFT_BUFFER, 0.0, SHIFT_CAP))
    if params.option_type == "put":
        return float(np.clip(threshold - SHIFT_BUFFER, -SHIFT_CAP, 0.0))
    raise ValueError("option_type must be 'call' or 'put'")


def simulate_state_value(spot, running_sum_before, day_index, params, rng, n_paths):
    m = future_count(day_index, params)
    total_fixings = params.n_fixings
    current_sum = running_sum_before + spot
    if m == 0:
        payoff = payoff_from_average(current_sum / total_fixings, params)
        return float(payoff), 0.0

    strike_adj = (total_fixings * params.strike - current_sum) / m
    if params.option_type == "call" and strike_adj <= 0.0:
        return float(exact_linear_value(spot, running_sum_before, day_index, params)), 0.0
    if params.option_type == "put" and strike_adj <= 0.0:
        return 0.0, 0.0

    theta = shift_amount(spot, running_sum_before, day_index, params)
    shift_vector = theta * bridge_shift_direction(m)
    dt = daily_dt(params)
    scale = m / total_fixings
    geo_exact = geometric_asian_exact(spot, running_sum_before, day_index, params)
    df = discount(day_index, params)

    count = 0
    sum_arith = 0.0
    sum_geo = 0.0
    sum_arith_sq = 0.0
    sum_geo_sq = 0.0
    sum_cross = 0.0
    remaining = int(n_paths)

    while remaining > 0:
        batch_paths = min(remaining, SIMULATION_BATCH_PATHS)
        half_paths = (batch_paths + 1) // 2
        base = rng.standard_normal((half_paths, m))
        normals = np.vstack((base + shift_vector, -base + shift_vector))[:batch_paths]
        likelihood_ratio = np.exp(-normals @ shift_vector + 0.5 * theta**2)

        increments = (
            (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
            + params.vol * sqrt(dt) * normals
        )
        cumulative_log_return = np.cumsum(increments, axis=1)
        future_spots = spot * np.exp(cumulative_log_return)
        future_arith_avg = np.sum(future_spots, axis=1) / m
        future_geo_avg = spot * np.exp(np.mean(cumulative_log_return, axis=1))

        if params.option_type == "call":
            arith_payoff = scale * np.maximum(future_arith_avg - strike_adj, 0.0)
            geo_payoff = scale * np.maximum(future_geo_avg - strike_adj, 0.0)
        else:
            arith_payoff = scale * np.maximum(strike_adj - future_arith_avg, 0.0)
            geo_payoff = scale * np.maximum(strike_adj - future_geo_avg, 0.0)

        discounted_arith = df * arith_payoff * likelihood_ratio
        discounted_geo = df * geo_payoff * likelihood_ratio

        count += batch_paths
        sum_arith += float(np.sum(discounted_arith))
        sum_geo += float(np.sum(discounted_geo))
        sum_arith_sq += float(np.sum(discounted_arith * discounted_arith))
        sum_geo_sq += float(np.sum(discounted_geo * discounted_geo))
        sum_cross += float(np.sum(discounted_arith * discounted_geo))
        remaining -= batch_paths

    mean_arith = sum_arith / count
    mean_geo = sum_geo / count
    geo_variance_num = sum_geo_sq - count * mean_geo * mean_geo
    covariance_num = sum_cross - count * mean_arith * mean_geo
    beta = covariance_num / geo_variance_num if geo_variance_num > 1e-14 else 0.0

    value = mean_arith - beta * (mean_geo - geo_exact)
    variance_num = (
        sum_arith_sq
        - count * mean_arith * mean_arith
        + beta * beta * geo_variance_num
        - 2.0 * beta * covariance_num
    )
    stderr = sqrt(max(variance_num / max(count - 1, 1), 0.0) / count)
    return value, stderr


def paths_per_state_from_budget(total_scenarios, n_states):
    return int(np.ceil(total_scenarios / max(n_states, 1)))


def build_labels(spots, running_sums, day_index, params, rng, n_paths):
    values = np.empty_like(spots, dtype=float)
    stderrs = np.empty_like(spots, dtype=float)
    for idx, (spot, running_sum) in enumerate(zip(spots, running_sums)):
        values[idx], stderrs[idx] = simulate_state_value(
            float(spot), float(running_sum), day_index, params, rng, n_paths
        )
    return values, stderrs


def scale_to_unit(values):
    values = np.asarray(values, dtype=float)
    lower = float(values.min())
    upper = float(values.max())
    if upper <= lower + 1e-12:
        return np.zeros_like(values), lower, upper
    return 2.0 * (values - lower) / (upper - lower) - 1.0, lower, upper


def apply_scale(values, lower, upper):
    values = np.asarray(values, dtype=float)
    if upper <= lower + 1e-12:
        return np.zeros_like(values)
    return np.clip(2.0 * (values - lower) / (upper - lower) - 1.0, -1.0, 1.0)


def ridge_solve(design, target):
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def fit_adjusted_hybrid_proxy(spots, running_sums, values, day_index, params):
    m = future_count(day_index, params)
    if m == 0:
        return lambda new_spots, new_running_sums: np.maximum(
            linear_values(new_spots, new_running_sums, day_index, params), 0.0
        )

    strike_adj = adjusted_strike(spots, running_sums, day_index, params)
    train_mask = strike_adj > 0.0
    x = adjusted_moneyness_coordinate(
        spots[train_mask], running_sums[train_mask], day_index, params
    )
    x_scaled, x_low, x_high = scale_to_unit(x)

    base = np.maximum(
        linear_values(spots[train_mask], running_sums[train_mask], day_index, params),
        0.0,
    )
    normalized_value = np.maximum(values[train_mask] / spots[train_mask], 0.0)
    normalized_time_value = np.maximum(
        (values[train_mask] - base) / spots[train_mask], 0.0
    )

    value_coeffs = ridge_solve(
        chebvander(x_scaled, HYBRID_VALUE_DEGREE),
        np.log(normalized_value + LOG_EPS),
    )
    time_coeffs = ridge_solve(
        chebvander(x_scaled, HYBRID_TIME_VALUE_DEGREE),
        np.log(normalized_time_value + LOG_EPS),
    )

    def predict(new_spots, new_running_sums):
        new_spots = np.asarray(new_spots, dtype=float)
        new_running_sums = np.asarray(new_running_sums, dtype=float)
        prediction = np.empty_like(new_spots, dtype=float)

        new_strike_adj = adjusted_strike(new_spots, new_running_sums, day_index, params)
        linear_tail = new_strike_adj <= 0.0
        prediction[linear_tail] = np.maximum(
            linear_values(
                new_spots[linear_tail],
                new_running_sums[linear_tail],
                day_index,
                params,
            ),
            0.0,
        )

        fitted = ~linear_tail
        new_x = apply_scale(
            adjusted_moneyness_coordinate(
                new_spots[fitted], new_running_sums[fitted], day_index, params
            ),
            x_low,
            x_high,
        )
        base_value = np.maximum(
            linear_values(new_spots[fitted], new_running_sums[fitted], day_index, params),
            0.0,
        )
        value_proxy = new_spots[fitted] * (
            np.exp(chebvander(new_x, HYBRID_VALUE_DEGREE) @ value_coeffs) - LOG_EPS
        )
        time_value_proxy = base_value + new_spots[fitted] * (
            np.exp(chebvander(new_x, HYBRID_TIME_VALUE_DEGREE) @ time_coeffs) - LOG_EPS
        )
        use_time_value = (base_value / np.maximum(new_spots[fitted], 1e-12)) > (
            HYBRID_INTRINSIC_SWITCH
        )
        prediction[fitted] = np.where(use_time_value, time_value_proxy, value_proxy)
        return np.maximum(prediction, 0.0)

    return predict


def score(prediction, benchmark, benchmark_stderr):
    abs_error = np.abs(prediction - benchmark)
    rel_error = abs_error / np.maximum(np.abs(benchmark), RELATIVE_ERROR_FLOOR)
    noise_ratio = abs_error / np.maximum(benchmark_stderr, 1e-12)
    return {
        "max_rel": float(np.max(rel_error)),
        "p99_rel": float(np.quantile(rel_error, 0.99)),
        "p95_rel": float(np.quantile(rel_error, 0.95)),
        "mae": float(np.mean(abs_error)),
        "max_abs": float(np.max(abs_error)),
        "median_error_over_benchmark_se": float(np.median(noise_ratio)),
    }


def run_day_index(params, day_index, rng):
    validation_spot, validation_sum = make_state_grid(
        day_index, params, VALIDATION_SPOT_POINTS, VALIDATION_AVG_POINTS
    )
    benchmark, benchmark_stderr = build_labels(
        validation_spot,
        validation_sum,
        day_index,
        params,
        rng,
        BENCHMARK_PATHS_PER_STATE,
    )

    train_spot, train_sum = make_adjusted_moneyness_grid(
        day_index, params, ADJUSTED_MONEYNESS_POINTS
    )
    train_paths_per_state = paths_per_state_from_budget(
        TRAIN_SCENARIOS_PER_FIT, len(train_spot)
    )
    train_values, train_stderr = build_labels(
        train_spot, train_sum, day_index, params, rng, train_paths_per_state
    )
    proxy = fit_adjusted_hybrid_proxy(train_spot, train_sum, train_values, day_index, params)
    prediction = proxy(validation_spot, validation_sum)

    return {
        "day_index": day_index,
        "time": day_index * daily_dt(params),
        "remaining_fixings_after_today": future_count(day_index, params),
        "method": METHOD_NAME,
        "detail": METHOD_DETAIL,
        "train_states": len(train_spot),
        "train_scenarios_per_fit_target": TRAIN_SCENARIOS_PER_FIT,
        "train_scenarios_used": train_paths_per_state * len(train_spot),
        "train_paths_per_state": train_paths_per_state,
        "validation_states": len(validation_spot),
        "benchmark_paths_per_state": BENCHMARK_PATHS_PER_STATE,
        **score(prediction, benchmark, benchmark_stderr),
        "avg_train_stderr": float(np.mean(train_stderr)),
        "avg_benchmark_stderr": float(np.mean(benchmark_stderr)),
    }


def write_results(rows):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    params = Params()
    rng = np.random.default_rng(params.seed + 503)
    rows = [run_day_index(params, day_index, rng) for day_index in TEST_DAY_INDICES]
    write_results(rows)

    print("Asian default proxy test")
    print(f"method: {METHOD_NAME} | {METHOD_DETAIL}")
    print(f"monthly fixings over {params.maturity:g}y: {params.n_fixings}")
    print(f"training scenarios/fit target: {TRAIN_SCENARIOS_PER_FIT:,}")
    print(f"benchmark paths/state: {BENCHMARK_PATHS_PER_STATE:,}")
    print()
    print("day    rem    max rel    p99 rel    MAE       max abs")
    print("---    ---    -------    -------    ---       -------")
    for row in rows:
        print(
            f"{row['day_index']:3d}    {row['remaining_fixings_after_today']:3d}    "
            f"{100.0 * row['max_rel']:7.3f}%   "
            f"{100.0 * row['p99_rel']:7.3f}%   "
            f"{row['mae']:8.6f}  {row['max_abs']:8.6f}"
        )
    print()
    print(f"results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
