import csv
from dataclasses import dataclass
from math import erf, exp, log, log1p, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.stats import norm, qmc

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


@dataclass(frozen=True)
class CliquetParams:
    s0: float = 100.0
    notional: float = 100.0
    rate: float = 0.05
    div_yield: float = 0.02
    vol: float = 0.20
    maturity: float = 1.0
    n_periods: int = 12
    local_floor: float = -0.02
    local_cap: float = 0.04
    global_floor: float = 0.0
    global_cap: float = 0.20
    seed: int = 23


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "Markdown"
    / "Cliquet"
    / "results"
    / "summary.md"
)
METHOD_CSV_PATH = OUTPUT_DIR / "cliquet_proxy_method_results.csv"
DETAIL_CSV_PATH = OUTPUT_DIR / "cliquet_proxy_validation_details.csv"

TEST_DAY_INDICES = [0, 3, 6, 9, 12]
UNIFORM_TRAIN_POINTS = 161
BOUNDARY_TRAIN_POINTS = 241
HALTON_TRAIN_POINTS = 241
VALIDATION_POINTS = 121
TRAIN_SCENARIOS_PER_FIT = 10_000_000
BENCHMARK_PATHS_PER_STATE = 524_288
SIMULATION_BATCH_PATHS = 131_072
RELATIVE_ERROR_FLOOR = 0.01
RIDGE = 1e-8
LOGIT_EPS = 1e-8


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


def period_dt(params):
    return params.maturity / params.n_periods


def remaining_periods(day_index, params):
    return params.n_periods - day_index


def discount(day_index, params):
    return exp(-params.rate * remaining_periods(day_index, params) * period_dt(params))


def payoff_from_accrued(total_return, params):
    return params.notional * np.clip(total_return, params.global_floor, params.global_cap)


def clipped_return_moments(params):
    dt = period_dt(params)
    mu = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
    sigma = params.vol * sqrt(dt)
    variance = sigma * sigma
    low = log1p(params.local_floor)
    high = log1p(params.local_cap)

    p_low = normal_cdf((low - mu) / sigma)
    p_high = 1.0 - normal_cdf((high - mu) / sigma)
    p_mid = 1.0 - p_low - p_high

    def exp_x_interval(power):
        shifted_mu = mu + power * variance
        low_z = (low - shifted_mu) / sigma
        high_z = (high - shifted_mu) / sigma
        return exp(power * mu + 0.5 * power * power * variance) * (
            normal_cdf(high_z) - normal_cdf(low_z)
        )

    e_x_mid = exp_x_interval(1.0)
    e_2x_mid = exp_x_interval(2.0)
    mid_return_mean = e_x_mid - p_mid
    mid_return_second = e_2x_mid - 2.0 * e_x_mid + p_mid

    mean = (
        params.local_floor * p_low
        + mid_return_mean
        + params.local_cap * p_high
    )
    second = (
        params.local_floor**2 * p_low
        + mid_return_second
        + params.local_cap**2 * p_high
    )
    return float(mean), float(max(second - mean * mean, 0.0))


def exact_tail_value(accrued, day_index, params):
    accrued = np.asarray(accrued, dtype=float)
    m = remaining_periods(day_index, params)
    if m == 0:
        return discount(day_index, params) * payoff_from_accrued(accrued, params)

    min_total = accrued + m * params.local_floor
    max_total = accrued + m * params.local_cap
    values = np.full_like(accrued, np.nan, dtype=float)
    values[min_total >= params.global_cap] = (
        discount(day_index, params) * params.notional * params.global_cap
    )
    values[max_total <= params.global_floor] = (
        discount(day_index, params) * params.notional * params.global_floor
    )
    return values


def exact_tail_mask(accrued, day_index, params):
    return np.isfinite(exact_tail_value(accrued, day_index, params))


def accrued_range(day_index, params):
    if day_index == 0:
        return 0.0, 0.0
    return day_index * params.local_floor, day_index * params.local_cap


def make_uniform_accrued_grid(day_index, params, n_points):
    low, high = accrued_range(day_index, params)
    if high <= low:
        return np.array([low], dtype=float)
    return np.linspace(low, high, n_points)


def transition_centers(day_index, params):
    m = remaining_periods(day_index, params)
    if m == 0:
        return []
    mean_coupon, variance_coupon = clipped_return_moments(params)
    mean_future = m * mean_coupon
    return [
        params.global_floor - mean_future,
        params.global_cap - mean_future,
    ]


def make_boundary_accrued_grid(day_index, params, n_points):
    low, high = accrued_range(day_index, params)
    if high <= low:
        return np.array([low], dtype=float)

    base_count = max(n_points // 2, 41)
    base = np.linspace(low, high, base_count)
    m = remaining_periods(day_index, params)
    _, variance_coupon = clipped_return_moments(params)
    width = max(sqrt(max(m * variance_coupon, 0.0)), (high - low) / 30.0, 1e-4)

    clusters = []
    for center in transition_centers(day_index, params):
        cluster = np.linspace(center - 3.0 * width, center + 3.0 * width, n_points // 4)
        clusters.append(np.clip(cluster, low, high))

    grid = np.unique(np.r_[base, *clusters])
    if len(grid) > n_points:
        pick = np.linspace(0, len(grid) - 1, n_points).round().astype(int)
        grid = grid[pick]
    return grid.astype(float)


def radical_inverse(index, base):
    result = 0.0
    fraction = 1.0 / base
    while index > 0:
        index, digit = divmod(index, base)
        result += digit * fraction
        fraction /= base
    return result


def make_halton_boundary_grid(day_index, params, n_points):
    low, high = accrued_range(day_index, params)
    if high <= low:
        return np.array([low], dtype=float)

    wide_count = n_points // 2
    boundary_count = n_points - wide_count
    wide = np.array(
        [low + radical_inverse(i + 1, 2) * (high - low) for i in range(wide_count)],
        dtype=float,
    )

    m = remaining_periods(day_index, params)
    _, variance_coupon = clipped_return_moments(params)
    width = max(sqrt(max(m * variance_coupon, 0.0)), (high - low) / 30.0, 1e-4)
    centers = transition_centers(day_index, params)
    if not centers:
        return np.unique(wide)

    boundary = []
    for i in range(boundary_count):
        center = centers[i % len(centers)]
        u = radical_inverse(i + 1, 3)
        boundary.append(np.clip(center + (2.0 * u - 1.0) * 3.0 * width, low, high))
    return np.unique(np.r_[wide, boundary]).astype(float)


def normalized_total_feature(accrued, day_index, params):
    accrued = np.asarray(accrued, dtype=float)
    m = remaining_periods(day_index, params)
    if m == 0:
        return np.zeros_like(accrued)
    mean_coupon, variance_coupon = clipped_return_moments(params)
    expected_total = accrued + m * mean_coupon
    midpoint = 0.5 * (params.global_floor + params.global_cap)
    stdev = sqrt(max(m * variance_coupon, 1e-12))
    return (expected_total - midpoint) / stdev


def cushion_features(accrued, day_index, params):
    accrued = np.asarray(accrued, dtype=float)
    m = remaining_periods(day_index, params)
    if m == 0:
        zeros = np.zeros_like(accrued)
        return zeros, zeros
    mean_coupon, variance_coupon = clipped_return_moments(params)
    expected_total = accrued + m * mean_coupon
    stdev = sqrt(max(m * variance_coupon, 1e-12))
    lower = (expected_total - params.global_floor) / stdev
    upper = (params.global_cap - expected_total) / stdev
    return lower, upper


def simulate_state_value(accrued, day_index, params, rng, n_paths):
    exact = exact_tail_value(np.array([accrued]), day_index, params)[0]
    if np.isfinite(exact):
        return float(exact), 0.0

    m = remaining_periods(day_index, params)
    dt = period_dt(params)
    drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
    vol_step = params.vol * sqrt(dt)
    df = discount(day_index, params)
    mean_coupon, _ = clipped_return_moments(params)
    control_exact = df * params.notional * m * mean_coupon

    count = 0
    sum_payoff = 0.0
    sum_control = 0.0
    sum_payoff_sq = 0.0
    sum_control_sq = 0.0
    sum_cross = 0.0
    seed = rng.integers(0, np.iinfo(np.int64).max)
    for normals in sobol_antithetic_batches(n_paths, m, seed, SIMULATION_BATCH_PATHS):
        batch_paths = len(normals)
        returns = np.exp(drift + vol_step * normals) - 1.0
        coupons = np.clip(returns, params.local_floor, params.local_cap)
        future_sum = np.sum(coupons, axis=1)
        discounted_payoff = df * payoff_from_accrued(accrued + future_sum, params)
        discounted_control = df * params.notional * future_sum

        count += batch_paths
        sum_payoff += float(np.sum(discounted_payoff))
        sum_control += float(np.sum(discounted_control))
        sum_payoff_sq += float(np.sum(discounted_payoff * discounted_payoff))
        sum_control_sq += float(np.sum(discounted_control * discounted_control))
        sum_cross += float(np.sum(discounted_payoff * discounted_control))

    mean_payoff = sum_payoff / count
    mean_control = sum_control / count
    control_var_num = sum_control_sq - count * mean_control * mean_control
    cov_num = sum_cross - count * mean_payoff * mean_control
    beta = cov_num / control_var_num if control_var_num > 1e-14 else 0.0
    value = mean_payoff - beta * (mean_control - control_exact)

    variance_num = (
        sum_payoff_sq
        - count * mean_payoff * mean_payoff
        + beta * beta * control_var_num
        - 2.0 * beta * cov_num
    )
    stderr = sqrt(max(variance_num / max(count - 1, 1), 0.0) / count)
    return float(value), float(stderr)


def paths_per_state_from_budget(total_scenarios, n_states):
    return qmc_path_count(np.ceil(total_scenarios / max(n_states, 1)))


def build_labels(accrued_grid, day_index, params, rng, n_paths):
    common_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    values = np.empty_like(accrued_grid, dtype=float)
    stderrs = np.empty_like(accrued_grid, dtype=float)
    for idx, accrued in enumerate(accrued_grid):
        state_rng = np.random.default_rng(common_seed)
        values[idx], stderrs[idx] = simulate_state_value(
            float(accrued), day_index, params, state_rng, n_paths
        )
    return values, stderrs


def scale_to_unit(values):
    values = np.asarray(values, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
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
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append(tx[:, i] * ty[:, j])
    return np.column_stack(cols)


def ridge_solve(design, target):
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def value_bounds(day_index, params):
    df = discount(day_index, params)
    lower = df * params.notional * params.global_floor
    upper = df * params.notional * params.global_cap
    return lower, upper


def logit(x):
    x = np.clip(x, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(x / (1.0 - x))


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


def fit_direct_chebyshev_1d(accrued, values, day_index, params, feature_name, degree):
    feature = feature_values(accrued, day_index, params, feature_name)
    scaled_feature, low, high = scale_to_unit(feature)
    coeffs = ridge_solve(chebvander(scaled_feature, degree), values)

    def predict(new_accrued):
        exact = exact_tail_value(new_accrued, day_index, params)
        output = np.empty_like(np.asarray(new_accrued, dtype=float))
        exact_mask = np.isfinite(exact)
        output[exact_mask] = exact[exact_mask]
        fitted = ~exact_mask
        new_feature = feature_values(new_accrued[fitted], day_index, params, feature_name)
        design = chebvander(apply_scale(new_feature, low, high), degree)
        output[fitted] = design @ coeffs
        lower, upper = value_bounds(day_index, params)
        return np.clip(output, lower, upper)

    return predict


def fit_logit_chebyshev_1d(accrued, values, day_index, params, feature_name, degree):
    exact = exact_tail_value(accrued, day_index, params)
    train_mask = ~np.isfinite(exact)
    if not np.any(train_mask):
        return lambda new_accrued: exact_tail_value(new_accrued, day_index, params)

    lower, upper = value_bounds(day_index, params)
    feature = feature_values(accrued[train_mask], day_index, params, feature_name)
    scaled_feature, low, high = scale_to_unit(feature)
    normalized = (values[train_mask] - lower) / max(upper - lower, 1e-12)
    coeffs = ridge_solve(chebvander(scaled_feature, degree), logit(normalized))

    def predict(new_accrued):
        new_accrued = np.asarray(new_accrued, dtype=float)
        exact_new = exact_tail_value(new_accrued, day_index, params)
        output = np.empty_like(new_accrued, dtype=float)
        exact_mask = np.isfinite(exact_new)
        output[exact_mask] = exact_new[exact_mask]
        fitted = ~exact_mask
        new_feature = feature_values(new_accrued[fitted], day_index, params, feature_name)
        design = chebvander(apply_scale(new_feature, low, high), degree)
        output[fitted] = lower + (upper - lower) * inv_logit(design @ coeffs)
        return np.clip(output, lower, upper)

    return predict


def fit_logit_chebyshev_2d(accrued, values, day_index, params, degree):
    exact = exact_tail_value(accrued, day_index, params)
    train_mask = ~np.isfinite(exact)
    if not np.any(train_mask):
        return lambda new_accrued: exact_tail_value(new_accrued, day_index, params)

    lower, upper = value_bounds(day_index, params)
    x, y = cushion_features(accrued[train_mask], day_index, params)
    x_scaled, x_low, x_high = scale_to_unit(x)
    y_scaled, y_low, y_high = scale_to_unit(y)
    design = chebyshev_terms_2d(x_scaled, y_scaled, degree)
    normalized = (values[train_mask] - lower) / max(upper - lower, 1e-12)
    coeffs = ridge_solve(design, logit(normalized))

    def predict(new_accrued):
        new_accrued = np.asarray(new_accrued, dtype=float)
        exact_new = exact_tail_value(new_accrued, day_index, params)
        output = np.empty_like(new_accrued, dtype=float)
        exact_mask = np.isfinite(exact_new)
        output[exact_mask] = exact_new[exact_mask]
        fitted = ~exact_mask
        new_x, new_y = cushion_features(new_accrued[fitted], day_index, params)
        new_design = chebyshev_terms_2d(
            apply_scale(new_x, x_low, x_high),
            apply_scale(new_y, y_low, y_high),
            degree,
        )
        output[fitted] = lower + (upper - lower) * inv_logit(new_design @ coeffs)
        return np.clip(output, lower, upper)

    return predict


def feature_values(accrued, day_index, params, feature_name):
    accrued = np.asarray(accrued, dtype=float)
    if feature_name == "accrued":
        return accrued
    if feature_name == "expected_total_z":
        return normalized_total_feature(accrued, day_index, params)
    raise ValueError(f"Unknown feature: {feature_name}")


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


def build_detail_rows(method_name, day_index, accrued, prediction, benchmark, stderr, params):
    rel_error = np.abs(prediction - benchmark) / np.maximum(
        np.abs(benchmark), RELATIVE_ERROR_FLOOR
    )
    z = normalized_total_feature(accrued, day_index, params)
    rows = []
    for idx, accrued_value in enumerate(accrued):
        rows.append(
            {
                "method": method_name,
                "day_index": day_index,
                "remaining_periods": remaining_periods(day_index, params),
                "accrued_return": float(accrued_value),
                "expected_total_z": float(z[idx]),
                "benchmark": float(benchmark[idx]),
                "proxy": float(prediction[idx]),
                "signed_error": float(prediction[idx] - benchmark[idx]),
                "abs_error": float(abs(prediction[idx] - benchmark[idx])),
                "relative_error": float(rel_error[idx]),
                "benchmark_stderr": float(stderr[idx]),
            }
        )
    return rows


def fit_method(method_name, accrued, values, day_index, params):
    if method_name == "direct_accrued_uniform_d7":
        return fit_direct_chebyshev_1d(accrued, values, day_index, params, "accrued", 7)
    if method_name == "logit_accrued_boundary_d9":
        return fit_logit_chebyshev_1d(accrued, values, day_index, params, "accrued", 9)
    if method_name == "logit_z_uniform_d9":
        return fit_logit_chebyshev_1d(
            accrued, values, day_index, params, "expected_total_z", 9
        )
    if method_name == "logit_z_boundary_d11":
        return fit_logit_chebyshev_1d(
            accrued, values, day_index, params, "expected_total_z", 11
        )
    if method_name == "logit_z_boundary_d19":
        return fit_logit_chebyshev_1d(
            accrued, values, day_index, params, "expected_total_z", 19
        )
    if method_name == "logit_z_halton_d11":
        return fit_logit_chebyshev_1d(
            accrued, values, day_index, params, "expected_total_z", 11
        )
    if method_name == "logit_cushion_2d_boundary_d5":
        return fit_logit_chebyshev_2d(accrued, values, day_index, params, 5)
    raise ValueError(f"Unknown method: {method_name}")


def method_description(method_name):
    descriptions = {
        "direct_accrued_uniform_d7": "Direct value Chebyshev degree 7 on raw accrued return, uniform grid",
        "logit_accrued_boundary_d9": "Bounded logit Chebyshev degree 9 on raw accrued return, boundary grid",
        "logit_z_uniform_d9": "Bounded logit Chebyshev degree 9 on expected-total z feature, uniform grid",
        "logit_z_boundary_d11": "Bounded logit Chebyshev degree 11 on expected-total z feature, boundary grid",
        "logit_z_boundary_d19": "Bounded logit Chebyshev degree 19 on expected-total z feature, boundary grid",
        "logit_z_halton_d11": "Bounded logit Chebyshev degree 11 on expected-total z feature, Halton/boundary grid",
        "logit_cushion_2d_boundary_d5": "Bounded logit 2D Chebyshev degree 5 on floor/cap cushion features, boundary grid",
    }
    return descriptions[method_name]


def run_experiment():
    params = CliquetParams()
    rng = np.random.default_rng(params.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    method_rows = []
    detail_rows = []

    train_grid_specs = {
        "uniform": (make_uniform_accrued_grid, UNIFORM_TRAIN_POINTS),
        "boundary": (make_boundary_accrued_grid, BOUNDARY_TRAIN_POINTS),
        "halton": (make_halton_boundary_grid, HALTON_TRAIN_POINTS),
    }
    method_grid = {
        "direct_accrued_uniform_d7": "uniform",
        "logit_accrued_boundary_d9": "boundary",
        "logit_z_uniform_d9": "uniform",
        "logit_z_boundary_d11": "boundary",
        "logit_z_boundary_d19": "boundary",
        "logit_z_halton_d11": "halton",
        "logit_cushion_2d_boundary_d5": "boundary",
    }

    plot_payloads = {}

    for day_index in TEST_DAY_INDICES:
        validation_grid = make_boundary_accrued_grid(day_index, params, VALIDATION_POINTS)
        benchmark, benchmark_stderr = build_labels(
            validation_grid, day_index, params, rng, BENCHMARK_PATHS_PER_STATE
        )

        label_cache = {}
        for grid_name, (grid_builder, point_count) in train_grid_specs.items():
            train_grid = grid_builder(day_index, params, point_count)
            train_paths = paths_per_state_from_budget(
                TRAIN_SCENARIOS_PER_FIT, len(train_grid)
            )
            train_values, train_stderr = build_labels(
                train_grid, day_index, params, rng, train_paths
            )
            label_cache[grid_name] = {
                "grid": train_grid,
                "values": train_values,
                "stderr": train_stderr,
                "paths": train_paths,
            }

        day_predictions = {}
        for method_name, grid_name in method_grid.items():
            labels = label_cache[grid_name]
            proxy = fit_method(
                method_name, labels["grid"], labels["values"], day_index, params
            )
            prediction = proxy(validation_grid)
            score = score_prediction(prediction, benchmark, benchmark_stderr)
            method_rows.append(
                {
                    "method": method_name,
                    "description": method_description(method_name),
                    "day_index": day_index,
                    "time": day_index * period_dt(params),
                    "remaining_periods": remaining_periods(day_index, params),
                    "train_grid": grid_name,
                    "train_states": len(labels["grid"]),
                    "train_scenarios_per_fit_target": TRAIN_SCENARIOS_PER_FIT,
                    "train_scenarios_used": labels["paths"] * len(labels["grid"]),
                    "train_paths_per_state": labels["paths"],
                    "validation_states": len(validation_grid),
                    "benchmark_paths_per_state": BENCHMARK_PATHS_PER_STATE,
                    **score,
                    "avg_train_stderr": float(np.mean(labels["stderr"])),
                    "avg_benchmark_stderr": float(np.mean(benchmark_stderr)),
                }
            )
            detail_rows.extend(
                build_detail_rows(
                    method_name,
                    day_index,
                    validation_grid,
                    prediction,
                    benchmark,
                    benchmark_stderr,
                    params,
                )
            )
            day_predictions[method_name] = prediction
            plot_payloads[(day_index, method_name)] = (
                validation_grid,
                prediction,
                benchmark,
            )
        print(f"completed day index {day_index}")

    best_method = best_overall_method(method_rows)
    for day_index in TEST_DAY_INDICES:
        grid, prediction, benchmark = plot_payloads[(day_index, best_method)]
        write_plot(day_index, grid, prediction, benchmark, params, best_method)

    return method_rows, detail_rows


def best_overall_method(method_rows):
    summary = summarize_by_method(method_rows)
    return min(
        summary,
        key=lambda method: (
            summary[method]["worst_max_rel"],
            summary[method]["avg_p99_rel"],
            summary[method]["avg_mae"],
        ),
    )


def summarize_by_method(rows):
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            "worst_max_rel": max(row["max_rel"] for row in subset),
            "avg_p99_rel": float(np.mean([row["p99_rel"] for row in subset])),
            "avg_p95_rel": float(np.mean([row["p95_rel"] for row in subset])),
            "avg_mae": float(np.mean([row["mae"] for row in subset])),
            "max_abs": max(row["max_abs"] for row in subset),
        }
    return summary


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(method_rows):
    params = CliquetParams()
    summary = summarize_by_method(method_rows)
    best_method = best_overall_method(method_rows)
    best_rows = [row for row in method_rows if row["method"] == best_method]

    lines = [
        "# Cliquet Option Proxy Experiment",
        "",
        "Monthly cliquet under GBM with 12 local return observations over 1 year.",
        "Payoff is notional times the globally floored/capped sum of locally floored/capped returns.",
        "",
        f"Local floor/cap: {100 * params.local_floor:.1f}% / {100 * params.local_cap:.1f}%.",
        f"Global floor/cap: {100 * params.global_floor:.1f}% / {100 * params.global_cap:.1f}%.",
        "",
        f"Each fitted proxy uses about {TRAIN_SCENARIOS_PER_FIT:,} MC training scenarios spread over",
        f"its training state grid. Benchmarks use {BENCHMARK_PATHS_PER_STATE:,} antithetic MC",
        "scenarios per validation state with an exact clipped-return-sum control variate.",
        "",
        "## Methods Compared",
        "",
        "| Method | Description | Worst Max % Error | Avg P99 % Error | Avg P95 % Error | Avg MAE | Max Abs Error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in sorted(
        summary,
        key=lambda item: (
            summary[item]["worst_max_rel"],
            summary[item]["avg_p99_rel"],
            summary[item]["avg_mae"],
        ),
    ):
        score = summary[method]
        lines.append(
            f"| `{method}` | {method_description(method)} | "
            f"{100 * score['worst_max_rel']:.3f}% | "
            f"{100 * score['avg_p99_rel']:.3f}% | "
            f"{100 * score['avg_p95_rel']:.3f}% | "
            f"{score['avg_mae']:.6f} | {score['max_abs']:.6f} |"
        )

    lines.extend(
        [
            "",
            f"## Best Overall Method By Day: `{best_method}`",
            "",
            "| Day Index | Remaining Periods | Max % Error | P99 % Error | P95 % Error | MAE | Max Abs Error |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(best_rows, key=lambda item: item["day_index"]):
        lines.append(
            f"| {row['day_index']} | {row['remaining_periods']} | "
            f"{100 * row['max_rel']:.3f}% | {100 * row['p99_rel']:.3f}% | "
            f"{100 * row['p95_rel']:.3f}% | {row['mae']:.6f} | {row['max_abs']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Main Takeaways",
            "",
            "- The natural Markov state is the accrued clipped return. Spot is not needed at reset dates",
            "  because future GBM returns are scale-invariant.",
            "- Raw value regression can work, but bounded-logit targets are more stable because the",
            "  cliquet payoff has known global floor/cap bounds.",
            "- The best feature in this run is the expected-total z-score: accrued return plus expected",
            "  future clipped return, normalized by future clipped-return volatility.",
            "- Exact tails are used when the remaining local floors/caps imply the global floor or cap",
            "  is already locked in.",
            "",
            "## Files",
            "",
            f"- Method summary CSV: `{METHOD_CSV_PATH.name}`",
            f"- Validation detail CSV: `{DETAIL_CSV_PATH.name}`",
            "- Best-method plots: `plots/cliquet_day_*.png`",
            "",
            "## Short Conclusion",
            "",
            f"The best overall method is `{best_method}`, with tested max error "
            f"{100 * summary[best_method]['worst_max_rel']:.3f}% and average p99 error "
            f"{100 * summary[best_method]['avg_p99_rel']:.3f}%.",
            "",
        ]
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
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


def write_plot(day_index, accrued, prediction, benchmark, params, method_name):
    if Image is None:
        return

    order = np.argsort(accrued)
    x = accrued[order]
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
        f"Cliquet proxy validation - day {day_index}, "
        f"{remaining_periods(day_index, params)} periods remaining",
        fill=(20, 20, 20),
    )
    draw.text((24, 42), f"method: {method_name}", fill=(70, 70, 70))
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
    image.save(PLOT_DIR / f"cliquet_day_{day_index:02d}_{method_name}.png")


def write_outputs(method_rows, detail_rows):
    write_csv(METHOD_CSV_PATH, method_rows)
    write_csv(DETAIL_CSV_PATH, detail_rows)
    write_summary(method_rows)


def main():
    method_rows, detail_rows = run_experiment()
    write_outputs(method_rows, detail_rows)
    summary = summarize_by_method(method_rows)

    print()
    print("Cliquet proxy experiment complete")
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
