import csv
from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.stats import norm, qmc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # Plotting is useful, but the numerical experiment should still run.
    Image = None
    ImageDraw = None
    ImageFont = None


@dataclass(frozen=True)
class AsianParams:
    s0: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    div_yield: float = 0.02
    vol: float = 0.20
    maturity: float = 1.0
    n_fixings: int = 12
    seed: int = 11
    option_type: str = "call"


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"
METHOD_CSV_PATH = OUTPUT_DIR / "asian_proxy_method_results.csv"
LEGACY_METHOD_CSV_PATH = OUTPUT_DIR / "asian_proxy_results.csv"
DETAIL_CSV_PATH = OUTPUT_DIR / "asian_proxy_validation_details.csv"

STATE_SPOT_POINTS = 15
STATE_AVG_POINTS = 11
VALIDATION_SPOT_POINTS = 9
VALIDATION_AVG_POINTS = 7
ADJUSTED_MONEYNESS_POINTS = 121
FORWARD_MONEYNESS_POINTS = 37
FORWARD_RATIO_POINTS = 11
FORWARD_LD_POINTS = 360
TRAIN_SCENARIOS_PER_FIT = 10_000_000
BENCHMARK_PATHS_PER_STATE = 524_288
SIMULATION_BATCH_PATHS = 131_072
TEST_DAY_INDICES = [0, 3, 6, 9, 11]

RAW_2D_DEGREE = 7
FORWARD_2D_DEGREE = 9
FORWARD_2D_VALUE_DEGREE = 9
FORWARD_2D_TIME_VALUE_DEGREE = 9
HYBRID_VALUE_DEGREE = 19
HYBRID_TIME_VALUE_DEGREE = 19
HYBRID_INTRINSIC_SWITCH = 0.05
RIDGE = 1e-8
LOG_EPS = 1e-12
RELATIVE_ERROR_FLOOR = 0.01
SHIFT_BUFFER = 0.5
SHIFT_CAP = 4.0


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


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
    spot = np.asarray(spot, dtype=float)
    tau = max(float(tau), daily_dt(params))
    return (
        np.log(spot / params.strike)
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


def expected_future_sum(spot, day_index, params):
    m = future_count(day_index, params)
    dt = daily_dt(params)
    growth_sum = sum(
        exp((params.rate - params.div_yield) * j * dt) for j in range(1, m + 1)
    )
    return np.asarray(spot, dtype=float) * growth_sum


def expected_future_growth_sum(day_index, params):
    m = future_count(day_index, params)
    dt = daily_dt(params)
    return sum(exp((params.rate - params.div_yield) * j * dt) for j in range(1, m + 1))


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


def state_coordinates(spot, running_sum_before, day_index, params):
    spot_coord = d1_like_spot(
        spot,
        future_count(day_index, params) * daily_dt(params),
        params,
    )
    if day_index == 0:
        avg_coord = np.zeros_like(np.asarray(spot, dtype=float))
    else:
        avg_before = np.maximum(
            np.asarray(running_sum_before, dtype=float) / day_index, 1e-12
        )
        avg_coord = np.log(avg_before / params.strike)
    return spot_coord, avg_coord


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
    d1_nodes = np.linspace(norm.ppf(0.001), norm.ppf(0.999), spot_points)
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
        return make_state_grid(day_index, params, STATE_SPOT_POINTS, STATE_AVG_POINTS)

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


def forward_state_coordinates(spot, running_sum_before, day_index, params):
    spot = np.asarray(spot, dtype=float)
    running_sum_before = np.asarray(running_sum_before, dtype=float)
    expected_average = (
        running_sum_before
        + spot
        + expected_future_sum(spot, day_index, params)
    ) / params.n_fixings
    forward_moneyness = np.log(np.maximum(expected_average, 1e-12) / params.strike)
    if day_index == 0:
        spot_history_ratio = np.log(np.maximum(spot, 1e-12) / params.strike)
    else:
        running_average = np.maximum(running_sum_before / day_index, 1e-12)
        spot_history_ratio = np.log(np.maximum(spot, 1e-12) / running_average)
    return forward_moneyness, spot_history_ratio


def states_from_forward_coordinates(
    forward_moneyness,
    spot_history_ratio,
    day_index,
    params,
):
    forward_average = params.strike * np.exp(np.asarray(forward_moneyness, dtype=float))
    growth_sum = expected_future_growth_sum(day_index, params)
    if day_index == 0:
        spot = forward_average * params.n_fixings / (1.0 + growth_sum)
        running_sum = np.zeros_like(spot)
        return spot, running_sum

    ratio = np.exp(np.asarray(spot_history_ratio, dtype=float))
    running_average = (
        forward_average
        * params.n_fixings
        / (day_index + ratio * (1.0 + growth_sum))
    )
    spot = ratio * running_average
    running_sum = day_index * running_average
    return spot, running_sum


def make_forward_moneyness_grid(day_index, params, moneyness_points, ratio_points):
    wide_x = np.linspace(log(0.55), log(1.55), moneyness_points)
    boundary_x = np.linspace(log(0.85), log(1.15), moneyness_points)
    x_nodes = np.unique(np.r_[wide_x, boundary_x])

    if day_index == 0:
        return states_from_forward_coordinates(x_nodes, np.zeros_like(x_nodes), day_index, params)

    wide_y = np.linspace(log(0.55), log(1.80), ratio_points)
    boundary_y = np.linspace(log(0.75), log(1.35), ratio_points)
    y_nodes = np.unique(np.r_[wide_y, boundary_y])

    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="xy")
    spot, running_sum = states_from_forward_coordinates(
        x_grid.ravel(), y_grid.ravel(), day_index, params
    )
    return spot.astype(float), running_sum.astype(float)


def radical_inverse(index, base):
    result = 0.0
    fraction = 1.0 / base
    while index > 0:
        index, digit = divmod(index, base)
        result += digit * fraction
        fraction /= base
    return result


def halton_sequence(n_points, dimension):
    primes = [2, 3, 5, 7, 11, 13, 17]
    if dimension > len(primes):
        raise ValueError("Not enough prime bases for requested Halton dimension")
    points = np.empty((n_points, dimension), dtype=float)
    for dim in range(dimension):
        base = primes[dim]
        for idx in range(n_points):
            points[idx, dim] = radical_inverse(idx + 1, base)
    return points


def make_forward_low_discrepancy_grid(day_index, params, n_points):
    wide_count = n_points // 2
    boundary_count = n_points - wide_count

    def map_box(points, x_low, x_high, y_low, y_high):
        x = x_low + points[:, 0] * (x_high - x_low)
        y = y_low + points[:, 1] * (y_high - y_low)
        return states_from_forward_coordinates(x, y, day_index, params)

    if day_index == 0:
        u_wide = halton_sequence(wide_count, 1)[:, 0]
        u_boundary = halton_sequence(boundary_count, 1)[:, 0]
        x_wide = log(0.55) + u_wide * (log(1.55) - log(0.55))
        x_boundary = log(0.85) + u_boundary * (log(1.15) - log(0.85))
        x = np.r_[x_wide, x_boundary]
        return states_from_forward_coordinates(x, np.zeros_like(x), day_index, params)

    wide_spot, wide_sum = map_box(
        halton_sequence(wide_count, 2),
        log(0.55),
        log(1.55),
        log(0.55),
        log(1.80),
    )
    boundary_spot, boundary_sum = map_box(
        halton_sequence(boundary_count, 2),
        log(0.85),
        log(1.15),
        log(0.75),
        log(1.35),
    )
    return (
        np.r_[wide_spot, boundary_spot].astype(float),
        np.r_[wide_sum, boundary_sum].astype(float),
    )


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
    seed = rng.integers(0, np.iinfo(np.int64).max)
    for base_normals in sobol_antithetic_batches(
        n_paths, m, seed, SIMULATION_BATCH_PATHS
    ):
        batch_paths = len(base_normals)
        normals = base_normals + shift_vector
        likelihood_ratio = np.exp(-normals @ shift_vector + 0.5 * theta**2)

        increments = (
            (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
            + params.vol * sqrt(dt) * normals
        )
        cumulative_log_return = np.cumsum(increments, axis=1)
        future_spots = spot * np.exp(cumulative_log_return)
        future_sum = np.sum(future_spots, axis=1)
        future_arith_avg = future_sum / m
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

    mean_arith = sum_arith / count
    mean_geo = sum_geo / count
    geo_variance_num = sum_geo_sq - count * mean_geo * mean_geo
    covariance_num = sum_cross - count * mean_arith * mean_geo
    if geo_variance_num > 1e-14:
        beta = covariance_num / geo_variance_num
    else:
        beta = 0.0

    value = mean_arith - beta * (mean_geo - geo_exact)
    variance_num = (
        sum_arith_sq
        - count * mean_arith * mean_arith
        + beta * beta * geo_variance_num
        - 2.0 * beta * covariance_num
    )
    sample_variance = max(variance_num / max(count - 1, 1), 0.0)
    stderr = sqrt(sample_variance / count)
    return value, stderr


def paths_per_state_from_budget(total_scenarios, n_states):
    return qmc_path_count(np.ceil(total_scenarios / max(n_states, 1)))


def build_labels(spots, running_sums, day_index, params, rng, n_paths):
    common_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    values = np.empty_like(spots, dtype=float)
    stderrs = np.empty_like(spots, dtype=float)
    for idx, (spot, running_sum) in enumerate(zip(spots, running_sums)):
        state_rng = np.random.default_rng(common_seed)
        values[idx], stderrs[idx] = simulate_state_value(
            float(spot), float(running_sum), day_index, params, state_rng, n_paths
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


def chebyshev_terms_2d(x, y, degree):
    tx = chebvander(x, degree)
    ty = chebvander(y, degree)
    columns = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            columns.append(tx[:, i] * ty[:, j])
    return np.column_stack(columns)


def ridge_solve(design, target):
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def fit_raw_2d_log_proxy(spots, running_sums, values, day_index, params):
    if future_count(day_index, params) == 0:
        return lambda new_spots, new_running_sums: np.maximum(
            linear_values(new_spots, new_running_sums, day_index, params), 0.0
        )

    sx, ax = state_coordinates(spots, running_sums, day_index, params)
    sx_scaled, sx_low, sx_high = scale_to_unit(sx)
    ax_scaled, ax_low, ax_high = scale_to_unit(ax)
    design = chebyshev_terms_2d(sx_scaled, ax_scaled, RAW_2D_DEGREE)
    coeffs = ridge_solve(design, np.log(np.maximum(values, 0.0) + 1e-10))

    def predict(new_spots, new_running_sums):
        new_sx, new_ax = state_coordinates(new_spots, new_running_sums, day_index, params)
        new_design = chebyshev_terms_2d(
            apply_scale(new_sx, sx_low, sx_high),
            apply_scale(new_ax, ax_low, ax_high),
            RAW_2D_DEGREE,
        )
        return np.maximum(np.exp(new_design @ coeffs) - 1e-10, 0.0)

    return predict


def fit_forward_2d_log_proxy(spots, running_sums, values, day_index, params):
    if future_count(day_index, params) == 0:
        return lambda new_spots, new_running_sums: np.maximum(
            linear_values(new_spots, new_running_sums, day_index, params), 0.0
        )

    fx, fy = forward_state_coordinates(spots, running_sums, day_index, params)
    fx_scaled, fx_low, fx_high = scale_to_unit(fx)
    fy_scaled, fy_low, fy_high = scale_to_unit(fy)
    design = chebyshev_terms_2d(fx_scaled, fy_scaled, FORWARD_2D_DEGREE)
    target = np.log(np.maximum(values / np.maximum(spots, 1e-12), 0.0) + LOG_EPS)
    coeffs = ridge_solve(design, target)

    def predict(new_spots, new_running_sums):
        new_spots = np.asarray(new_spots, dtype=float)
        new_fx, new_fy = forward_state_coordinates(
            new_spots, new_running_sums, day_index, params
        )
        new_design = chebyshev_terms_2d(
            apply_scale(new_fx, fx_low, fx_high),
            apply_scale(new_fy, fy_low, fy_high),
            FORWARD_2D_DEGREE,
        )
        return np.maximum(new_spots * (np.exp(new_design @ coeffs) - LOG_EPS), 0.0)

    return predict


def fit_forward_2d_hybrid_proxy(spots, running_sums, values, day_index, params):
    m = future_count(day_index, params)
    if m == 0:
        return lambda new_spots, new_running_sums: np.maximum(
            linear_values(new_spots, new_running_sums, day_index, params), 0.0
        )

    strike_adj = adjusted_strike(spots, running_sums, day_index, params)
    train_mask = strike_adj > 0.0
    fx, fy = forward_state_coordinates(
        spots[train_mask], running_sums[train_mask], day_index, params
    )
    fx_scaled, fx_low, fx_high = scale_to_unit(fx)
    fy_scaled, fy_low, fy_high = scale_to_unit(fy)
    value_design = chebyshev_terms_2d(
        fx_scaled, fy_scaled, FORWARD_2D_VALUE_DEGREE
    )
    time_design = chebyshev_terms_2d(
        fx_scaled, fy_scaled, FORWARD_2D_TIME_VALUE_DEGREE
    )
    base = np.maximum(
        linear_values(spots[train_mask], running_sums[train_mask], day_index, params),
        0.0,
    )
    normalized_value = np.maximum(
        values[train_mask] / np.maximum(spots[train_mask], 1e-12), 0.0
    )
    normalized_time_value = np.maximum(
        (values[train_mask] - base) / np.maximum(spots[train_mask], 1e-12), 0.0
    )
    value_coeffs = ridge_solve(value_design, np.log(normalized_value + LOG_EPS))
    time_coeffs = ridge_solve(time_design, np.log(normalized_time_value + LOG_EPS))

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
        new_fx, new_fy = forward_state_coordinates(
            new_spots[fitted], new_running_sums[fitted], day_index, params
        )
        scaled_fx = apply_scale(new_fx, fx_low, fx_high)
        scaled_fy = apply_scale(new_fy, fy_low, fy_high)
        base_value = np.maximum(
            linear_values(new_spots[fitted], new_running_sums[fitted], day_index, params),
            0.0,
        )
        value_proxy = new_spots[fitted] * (
            np.exp(
                chebyshev_terms_2d(scaled_fx, scaled_fy, FORWARD_2D_VALUE_DEGREE)
                @ value_coeffs
            )
            - LOG_EPS
        )
        time_value_proxy = base_value + new_spots[fitted] * (
            np.exp(
                chebyshev_terms_2d(
                    scaled_fx, scaled_fy, FORWARD_2D_TIME_VALUE_DEGREE
                )
                @ time_coeffs
            )
            - LOG_EPS
        )
        use_time_value = (base_value / np.maximum(new_spots[fitted], 1e-12)) > (
            HYBRID_INTRINSIC_SWITCH
        )
        prediction[fitted] = np.where(use_time_value, time_value_proxy, value_proxy)
        return np.maximum(prediction, 0.0)

    return predict


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

    value_design = chebvander(x_scaled, HYBRID_VALUE_DEGREE)
    time_design = chebvander(x_scaled, HYBRID_TIME_VALUE_DEGREE)
    value_coeffs = ridge_solve(value_design, np.log(normalized_value + LOG_EPS))
    time_coeffs = ridge_solve(time_design, np.log(normalized_time_value + LOG_EPS))

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
        log_value = chebvander(new_x, HYBRID_VALUE_DEGREE) @ value_coeffs
        log_time_value = chebvander(new_x, HYBRID_TIME_VALUE_DEGREE) @ time_coeffs
        value_proxy = new_spots[fitted] * (np.exp(log_value) - LOG_EPS)
        time_value_proxy = base_value + new_spots[fitted] * (
            np.exp(log_time_value) - LOG_EPS
        )

        use_time_value = (base_value / np.maximum(new_spots[fitted], 1e-12)) > (
            HYBRID_INTRINSIC_SWITCH
        )
        prediction[fitted] = np.where(use_time_value, time_value_proxy, value_proxy)
        return np.maximum(prediction, 0.0)

    return predict


def score_prediction(prediction, benchmark, benchmark_stderr):
    signed_error = prediction - benchmark
    abs_error = np.abs(signed_error)
    rel_error = abs_error / np.maximum(np.abs(benchmark), RELATIVE_ERROR_FLOOR)
    noise_ratio = abs_error / np.maximum(benchmark_stderr, 1e-12)
    return {
        "max_rel": float(np.max(rel_error)),
        "p99_rel": float(np.quantile(rel_error, 0.99)),
        "p95_rel": float(np.quantile(rel_error, 0.95)),
        "mae": float(np.mean(abs_error)),
        "max_abs": float(np.max(abs_error)),
        "median_noise_ratio": float(np.median(noise_ratio)),
    }


def build_detail_rows(
    method_name,
    day_index,
    spots,
    running_sums,
    prediction,
    benchmark,
    benchmark_stderr,
    params,
):
    linear_baseline = np.maximum(linear_values(spots, running_sums, day_index, params), 0.0)
    rel_error = np.abs(prediction - benchmark) / np.maximum(
        np.abs(benchmark), RELATIVE_ERROR_FLOOR
    )
    adjusted_d1 = adjusted_moneyness_coordinate(spots, running_sums, day_index, params)
    strike_adj = adjusted_strike(spots, running_sums, day_index, params)
    rows = []
    for idx, (spot, running_sum) in enumerate(zip(spots, running_sums)):
        avg_before = 0.0 if day_index == 0 else float(running_sum / day_index)
        rows.append(
            {
                "method": method_name,
                "day_index": day_index,
                "remaining_fixings_after_today": future_count(day_index, params),
                "spot": float(spot),
                "running_sum_before": float(running_sum),
                "running_average_before": avg_before,
                "adjusted_strike": float(strike_adj[idx]),
                "adjusted_d1": float(adjusted_d1[idx]),
                "linear_baseline": float(linear_baseline[idx]),
                "benchmark": float(benchmark[idx]),
                "proxy": float(prediction[idx]),
                "signed_error": float(prediction[idx] - benchmark[idx]),
                "abs_error": float(abs(prediction[idx] - benchmark[idx])),
                "relative_error": float(rel_error[idx]),
                "benchmark_stderr": float(benchmark_stderr[idx]),
            }
        )
    return rows


def run_experiment():
    params = AsianParams()
    rng = np.random.default_rng(params.seed)
    method_rows = []
    detail_rows = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for day_index in TEST_DAY_INDICES:
        validation_spot, validation_sum = make_state_grid(
            day_index, params, VALIDATION_SPOT_POINTS, VALIDATION_AVG_POINTS
        )
        benchmark_values, benchmark_stderr = build_labels(
            validation_spot,
            validation_sum,
            day_index,
            params,
            rng,
            BENCHMARK_PATHS_PER_STATE,
        )

        method_specs = []

        raw_train_spot, raw_train_sum = make_state_grid(
            day_index, params, STATE_SPOT_POINTS, STATE_AVG_POINTS
        )
        raw_train_paths_per_state = paths_per_state_from_budget(
            TRAIN_SCENARIOS_PER_FIT, len(raw_train_spot)
        )
        raw_train_values, raw_train_stderr = build_labels(
            raw_train_spot,
            raw_train_sum,
            day_index,
            params,
            rng,
            raw_train_paths_per_state,
        )
        raw_proxy = fit_raw_2d_log_proxy(
            raw_train_spot, raw_train_sum, raw_train_values, day_index, params
        )
        method_specs.append(
            {
                "method": "raw_2d_log_chebyshev",
                "proxy": raw_proxy,
                "train_states": len(raw_train_spot),
                "train_paths_per_state": raw_train_paths_per_state,
                "train_scenarios_used": raw_train_paths_per_state * len(raw_train_spot),
                "avg_train_stderr": float(np.mean(raw_train_stderr)),
                "description": f"2D log Chebyshev degree {RAW_2D_DEGREE}",
            }
        )

        forward_train_spot, forward_train_sum = make_forward_moneyness_grid(
            day_index, params, FORWARD_MONEYNESS_POINTS, FORWARD_RATIO_POINTS
        )
        forward_train_paths_per_state = paths_per_state_from_budget(
            TRAIN_SCENARIOS_PER_FIT, len(forward_train_spot)
        )
        forward_train_values, forward_train_stderr = build_labels(
            forward_train_spot,
            forward_train_sum,
            day_index,
            params,
            rng,
            forward_train_paths_per_state,
        )
        forward_log_proxy = fit_forward_2d_log_proxy(
            forward_train_spot,
            forward_train_sum,
            forward_train_values,
            day_index,
            params,
        )
        method_specs.append(
            {
                "method": "forward_grid_2d_log_chebyshev",
                "proxy": forward_log_proxy,
                "train_states": len(forward_train_spot),
                "train_paths_per_state": forward_train_paths_per_state,
                "train_scenarios_used": (
                    forward_train_paths_per_state * len(forward_train_spot)
                ),
                "avg_train_stderr": float(np.mean(forward_train_stderr)),
                "description": (
                    f"Forward-coordinate 2D log Chebyshev degree {FORWARD_2D_DEGREE}, "
                    "boundary-enriched tensor grid"
                ),
            }
        )
        forward_hybrid_proxy = fit_forward_2d_hybrid_proxy(
            forward_train_spot,
            forward_train_sum,
            forward_train_values,
            day_index,
            params,
        )
        method_specs.append(
            {
                "method": "forward_grid_2d_hybrid",
                "proxy": forward_hybrid_proxy,
                "train_states": len(forward_train_spot),
                "train_paths_per_state": forward_train_paths_per_state,
                "train_scenarios_used": (
                    forward_train_paths_per_state * len(forward_train_spot)
                ),
                "avg_train_stderr": float(np.mean(forward_train_stderr)),
                "description": (
                    f"Forward-coordinate 2D hybrid degrees "
                    f"{FORWARD_2D_VALUE_DEGREE}/{FORWARD_2D_TIME_VALUE_DEGREE}, "
                    "boundary-enriched tensor grid"
                ),
            }
        )

        ld_train_spot, ld_train_sum = make_forward_low_discrepancy_grid(
            day_index, params, FORWARD_LD_POINTS
        )
        ld_train_paths_per_state = paths_per_state_from_budget(
            TRAIN_SCENARIOS_PER_FIT, len(ld_train_spot)
        )
        ld_train_values, ld_train_stderr = build_labels(
            ld_train_spot,
            ld_train_sum,
            day_index,
            params,
            rng,
            ld_train_paths_per_state,
        )
        ld_log_proxy = fit_forward_2d_log_proxy(
            ld_train_spot, ld_train_sum, ld_train_values, day_index, params
        )
        method_specs.append(
            {
                "method": "forward_halton_2d_log_chebyshev",
                "proxy": ld_log_proxy,
                "train_states": len(ld_train_spot),
                "train_paths_per_state": ld_train_paths_per_state,
                "train_scenarios_used": ld_train_paths_per_state * len(ld_train_spot),
                "avg_train_stderr": float(np.mean(ld_train_stderr)),
                "description": (
                    f"Forward-coordinate 2D log Chebyshev degree {FORWARD_2D_DEGREE}, "
                    "Halton global/boundary state sampling"
                ),
            }
        )
        ld_hybrid_proxy = fit_forward_2d_hybrid_proxy(
            ld_train_spot, ld_train_sum, ld_train_values, day_index, params
        )
        method_specs.append(
            {
                "method": "forward_halton_2d_hybrid",
                "proxy": ld_hybrid_proxy,
                "train_states": len(ld_train_spot),
                "train_paths_per_state": ld_train_paths_per_state,
                "train_scenarios_used": ld_train_paths_per_state * len(ld_train_spot),
                "avg_train_stderr": float(np.mean(ld_train_stderr)),
                "description": (
                    f"Forward-coordinate 2D hybrid degrees "
                    f"{FORWARD_2D_VALUE_DEGREE}/{FORWARD_2D_TIME_VALUE_DEGREE}, "
                    "Halton global/boundary state sampling"
                ),
            }
        )

        hybrid_train_spot, hybrid_train_sum = make_adjusted_moneyness_grid(
            day_index, params, ADJUSTED_MONEYNESS_POINTS
        )
        hybrid_train_paths_per_state = paths_per_state_from_budget(
            TRAIN_SCENARIOS_PER_FIT, len(hybrid_train_spot)
        )
        hybrid_train_values, hybrid_train_stderr = build_labels(
            hybrid_train_spot,
            hybrid_train_sum,
            day_index,
            params,
            rng,
            hybrid_train_paths_per_state,
        )
        hybrid_proxy = fit_adjusted_hybrid_proxy(
            hybrid_train_spot, hybrid_train_sum, hybrid_train_values, day_index, params
        )
        method_specs.append(
            {
                "method": "adjusted_moneyness_hybrid",
                "proxy": hybrid_proxy,
                "train_states": len(hybrid_train_spot),
                "train_paths_per_state": hybrid_train_paths_per_state,
                "train_scenarios_used": (
                    hybrid_train_paths_per_state * len(hybrid_train_spot)
                ),
                "avg_train_stderr": float(np.mean(hybrid_train_stderr)),
                "description": (
                    f"1D adjusted-moneyness hybrid: log value degree "
                    f"{HYBRID_VALUE_DEGREE}, log time-value degree "
                    f"{HYBRID_TIME_VALUE_DEGREE}"
                ),
            }
        )

        for method_spec in method_specs:
            prediction = method_spec["proxy"](validation_spot, validation_sum)
            score = score_prediction(prediction, benchmark_values, benchmark_stderr)
            method_rows.append(
                {
                    "method": method_spec["method"],
                    "description": method_spec["description"],
                    "day_index": day_index,
                    "calendar_time": day_index * daily_dt(params),
                    "remaining_fixings_after_today": future_count(day_index, params),
                    "train_states": method_spec["train_states"],
                    "validation_states": len(validation_spot),
                    "train_scenarios_per_fit_target": TRAIN_SCENARIOS_PER_FIT,
                    "train_scenarios_used": method_spec["train_scenarios_used"],
                    "train_paths_per_state": method_spec["train_paths_per_state"],
                    "benchmark_paths_per_state": BENCHMARK_PATHS_PER_STATE,
                    **score,
                    "avg_train_stderr": method_spec["avg_train_stderr"],
                    "avg_benchmark_stderr": float(np.mean(benchmark_stderr)),
                }
            )
            detail_rows.extend(
                build_detail_rows(
                    method_spec["method"],
                    day_index,
                    validation_spot,
                    validation_sum,
                    prediction,
                    benchmark_values,
                    benchmark_stderr,
                    params,
                )
            )

            if method_spec["method"] == "adjusted_moneyness_hybrid":
                write_plot(
                    day_index,
                    validation_spot,
                    validation_sum,
                    prediction,
                    benchmark_values,
                    params,
                )

        print(f"completed day index {day_index}")

    return method_rows, detail_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_method(rows):
    methods = sorted({row["method"] for row in rows})
    summary = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            "worst_max_rel": max(row["max_rel"] for row in subset),
            "avg_p99_rel": float(np.mean([row["p99_rel"] for row in subset])),
            "avg_p95_rel": float(np.mean([row["p95_rel"] for row in subset])),
            "avg_mae": float(np.mean([row["mae"] for row in subset])),
            "max_abs": max(row["max_abs"] for row in subset),
        }
    return summary


def write_summary(method_rows):
    method_summary = summarize_by_method(method_rows)
    method_labels = {
        "raw_2d_log_chebyshev": "Naive 2D extension",
        "forward_grid_2d_log_chebyshev": "Forward 2D log, tensor",
        "forward_grid_2d_hybrid": "Forward 2D hybrid, tensor",
        "forward_halton_2d_log_chebyshev": "Forward 2D log, Halton",
        "forward_halton_2d_hybrid": "Forward 2D hybrid, Halton",
        "adjusted_moneyness_hybrid": "Adjusted-moneyness hybrid",
    }
    best_method = min(
        method_summary,
        key=lambda method: (
            method_summary[method]["worst_max_rel"],
            method_summary[method]["avg_p99_rel"],
            method_summary[method]["avg_mae"],
        ),
    )
    best_rows = [row for row in method_rows if row["method"] == best_method]

    lines = [
        "# Asian Option Proxy Experiment",
        "",
        "Monthly arithmetic Asian call under GBM with 12 fixings over 1 year.",
        "State is `(spot today, running sum before today)`.",
        "",
        f"No arithmetic Asian closed form is used. Each fitted daily proxy uses about "
        f"{TRAIN_SCENARIOS_PER_FIT:,} shifted-antithetic MC training scenarios spread over",
        f"that fit's training grid. Benchmarks use {BENCHMARK_PATHS_PER_STATE:,} shifted-antithetic",
        "MC scenarios per validation state, with a discrete geometric Asian control variate.",
        "",
        "Relative errors below use a denominator floor of `0.01`, so near-zero prices do not dominate",
        "the percentages purely through division by a tiny value.",
        "",
        "## Methods Compared",
        "",
        "| Method | Description | Worst Max % Error | Avg P99 % Error | Avg P95 % Error | Avg MAE | Max Abs Error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for method in sorted(
        method_summary,
        key=lambda item: (
            method_summary[item]["worst_max_rel"],
            method_summary[item]["avg_p99_rel"],
            method_summary[item]["avg_mae"],
        ),
    ):
        label = method_labels.get(method, method)
        score = method_summary[method]
        description = next(row["description"] for row in method_rows if row["method"] == method)
        lines.append(
            f"| {label} | {description} | "
            f"{100.0 * score['worst_max_rel']:.3f}% | "
            f"{100.0 * score['avg_p99_rel']:.3f}% | "
            f"{100.0 * score['avg_p95_rel']:.3f}% | "
            f"{score['avg_mae']:.6f} | "
            f"{score['max_abs']:.6f} |"
        )

    lines.extend(
        [
            "",
            f"## Best Overall Method By Day: {method_labels.get(best_method, best_method)}",
            "",
            "| Day Index | Remaining Fixings | Max % Error | P99 % Error | P95 % Error | MAE | Max Abs Error |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(best_rows, key=lambda item: item["day_index"]):
        lines.append(
            f"| {row['day_index']} | {row['remaining_fixings_after_today']} | "
            f"{100.0 * row['max_rel']:.3f}% | "
            f"{100.0 * row['p99_rel']:.3f}% | "
            f"{100.0 * row['p95_rel']:.3f}% | "
            f"{row['mae']:.6f} | {row['max_abs']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## What Changed Versus The Naive 2D Proxy",
            "",
            "- The raw 2D log-Chebyshev extension works at inception but fails later because the payoff",
            "  boundary is diagonal in `(spot, running average)` and the ITM wing becomes almost exactly",
            "  linear.",
            "- The forward-coordinate methods sample and fit in expected-average moneyness plus",
            "  spot/history ratio, which directly tests enriched OTM/ITM state sampling.",
            "- The Halton variants use low-discrepancy state sampling in those same forward coordinates,",
            "  with half the points in a global box and half near the payoff boundary.",
            "- For fixed day under GBM, the Asian continuation payoff can be rewritten with an adjusted",
            "  strike: `K_adj = (N K - running_sum_before - spot) / future_fixings`.",
            "- GBM scaling implies the continuation value can be represented by",
            "  `spot * f(K_adj / spot)` for each remaining tenor, so the best proxy is fitted in",
            "  adjusted moneyness rather than raw state coordinates.",
            "- The hybrid uses a log-price fit near/OTM and switches to `linear baseline + log time value`",
            f"  once the linear baseline exceeds {100.0 * HYBRID_INTRINSIC_SWITCH:.1f}% of spot.",
            "- The exact terminal payoff and the `K_adj <= 0` linear region are handled analytically.",
            "",
            "## Files",
            "",
            f"- Method summary CSV: `{METHOD_CSV_PATH.name}`",
            f"- Validation detail CSV: `{DETAIL_CSV_PATH.name}`",
            "- Hybrid plots: `plots/asian_day_*.png`",
            "",
        ]
    )

    lines.extend(
        [
            "## Short Conclusion",
            "",
            f"The best overall method in this run is `{best_method}`, with tested max error",
            f"{100.0 * method_summary[best_method]['worst_max_rel']:.3f}% and average p99 error",
            f"{100.0 * method_summary[best_method]['avg_p99_rel']:.3f}%.",
            "",
        ]
    )

    SUMMARY_PATH.write_text("\n".join(lines))


def draw_panel(draw, x0, y0, x1, y1, title, xs, series, colors):
    draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210), width=1)
    draw.text((x0 + 8, y0 + 6), title, fill=(35, 35, 35))
    all_values = np.concatenate([np.asarray(values, dtype=float) for values in series])
    finite = all_values[np.isfinite(all_values)]
    ymin = float(np.min(finite)) if len(finite) else 0.0
    ymax = float(np.max(finite)) if len(finite) else 1.0
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    padding = 0.08 * (ymax - ymin)
    ymin -= padding
    ymax += padding
    xmin = float(np.min(xs))
    xmax = float(np.max(xs))
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0

    def point(x, y):
        px = x0 + 46 + (x - xmin) / (xmax - xmin) * (x1 - x0 - 64)
        py = y1 - 24 - (y - ymin) / (ymax - ymin) * (y1 - y0 - 54)
        return int(px), int(py)

    zero_y = None
    if ymin < 0.0 < ymax:
        zero_y = point(xmin, 0.0)[1]
        draw.line((x0 + 38, zero_y, x1 - 12, zero_y), fill=(225, 225, 225), width=1)

    for values, color in zip(series, colors):
        pts = [point(float(x), float(y)) for x, y in zip(xs, values)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
        for px, py in pts:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)

    draw.text((x0 + 8, y1 - 18), f"{ymin:.3g}", fill=(90, 90, 90))
    draw.text((x1 - 70, y0 + 22), f"{ymax:.3g}", fill=(90, 90, 90))
    if zero_y is not None:
        draw.text((x0 + 8, zero_y - 8), "0", fill=(90, 90, 90))


def write_plot(day_index, spots, running_sums, prediction, benchmark, params):
    if Image is None:
        return

    x = adjusted_moneyness_coordinate(spots, running_sums, day_index, params)
    order = np.argsort(x)
    x = x[order]
    benchmark = benchmark[order]
    prediction = prediction[order]
    signed_error = prediction - benchmark
    rel_error = 100.0 * np.abs(signed_error) / np.maximum(
        np.abs(benchmark), RELATIVE_ERROR_FLOOR
    )

    width, height = 1200, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (24, 18),
        f"Asian hybrid proxy validation - day {day_index}, "
        f"{future_count(day_index, params)} fixings remaining after today",
        fill=(20, 20, 20),
    )
    draw.text((24, 42), "x-axis: adjusted d1 moneyness", fill=(70, 70, 70))
    draw.text((900, 42), "black: benchmark, red: proxy", fill=(70, 70, 70))

    draw_panel(
        draw,
        24,
        80,
        width - 24,
        360,
        "Value",
        x,
        [benchmark, prediction],
        [(20, 20, 20), (210, 30, 40)],
    )
    draw_panel(
        draw,
        24,
        390,
        width - 24,
        620,
        "Signed error: proxy - benchmark",
        x,
        [signed_error],
        [(30, 130, 70)],
    )
    draw_panel(
        draw,
        24,
        650,
        width - 24,
        870,
        "Floored relative error (%)",
        x,
        [rel_error],
        [(30, 80, 180)],
    )

    image.save(PLOT_DIR / f"asian_day_{day_index:02d}_hybrid.png")


def write_outputs(method_rows, detail_rows):
    write_csv(METHOD_CSV_PATH, method_rows)
    write_csv(LEGACY_METHOD_CSV_PATH, method_rows)
    write_csv(DETAIL_CSV_PATH, detail_rows)
    write_summary(method_rows)


def main():
    method_rows, detail_rows = run_experiment()
    write_outputs(method_rows, detail_rows)
    summary = summarize_by_method(method_rows)

    print()
    print("Asian proxy experiment complete")
    print(f"method CSV written to: {METHOD_CSV_PATH}")
    print(f"detail CSV written to: {DETAIL_CSV_PATH}")
    print(f"summary written to: {SUMMARY_PATH}")
    if Image is not None:
        print(f"plots written to: {PLOT_DIR}")
    print()
    print("method                              worst max rel    avg p99 rel    avg MAE")
    print("---------------------------------   -------------    -----------    -------")
    for method in sorted(
        summary,
        key=lambda item: (
            summary[item]["worst_max_rel"],
            summary[item]["avg_p99_rel"],
            summary[item]["avg_mae"],
        ),
    ):
        score = summary[method]
        print(
            f"{method:33s}   "
            f"{100.0 * score['worst_max_rel']:13.3f}%   "
            f"{100.0 * score['avg_p99_rel']:11.3f}%   "
            f"{score['avg_mae']:7.5f}"
        )


if __name__ == "__main__":
    main()
