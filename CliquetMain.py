import csv
from dataclasses import dataclass
from math import erf, exp, log, log1p, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander


@dataclass(frozen=True)
class Params:
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


METHOD_NAME = "bounded-logit z Chebyshev"
METHOD_DETAIL = "expected-total z feature, boundary grid, degree=19"
TEST_DAY_INDICES = [0, 3, 6, 9, 12]
TRAIN_POINTS = 241
VALIDATION_POINTS = 121
TRAIN_SCENARIOS_PER_FIT = 10_000_000
BENCHMARK_PATHS_PER_STATE = 500_000
SIMULATION_BATCH_PATHS = 100_000
DEGREE = 19
RIDGE = 1e-8
LOGIT_EPS = 1e-8
RELATIVE_ERROR_FLOOR = 0.01
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "CliquetOptExperiment"
    / "default_run"
    / "cliquet_default_results.csv"
)


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


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


def value_bounds(day_index, params):
    df = discount(day_index, params)
    return df * params.notional * params.global_floor, df * params.notional * params.global_cap


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


def accrued_range(day_index, params):
    if day_index == 0:
        return 0.0, 0.0
    return day_index * params.local_floor, day_index * params.local_cap


def transition_centers(day_index, params):
    m = remaining_periods(day_index, params)
    if m == 0:
        return []
    mean_coupon, _ = clipped_return_moments(params)
    mean_future = m * mean_coupon
    return [params.global_floor - mean_future, params.global_cap - mean_future]


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


def expected_total_z(accrued, day_index, params):
    accrued = np.asarray(accrued, dtype=float)
    m = remaining_periods(day_index, params)
    if m == 0:
        return np.zeros_like(accrued)
    mean_coupon, variance_coupon = clipped_return_moments(params)
    expected_total = accrued + m * mean_coupon
    midpoint = 0.5 * (params.global_floor + params.global_cap)
    stdev = sqrt(max(m * variance_coupon, 1e-12))
    return (expected_total - midpoint) / stdev


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
    remaining = int(n_paths)

    while remaining > 0:
        batch_paths = min(remaining, SIMULATION_BATCH_PATHS)
        half_paths = (batch_paths + 1) // 2
        base = rng.standard_normal((half_paths, m))
        normals = np.vstack((base, -base))[:batch_paths]
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
        remaining -= batch_paths

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
    return int(np.ceil(total_scenarios / max(n_states, 1)))


def build_labels(accrued_grid, day_index, params, rng, n_paths):
    values = np.empty_like(accrued_grid, dtype=float)
    stderrs = np.empty_like(accrued_grid, dtype=float)
    for idx, accrued in enumerate(accrued_grid):
        values[idx], stderrs[idx] = simulate_state_value(
            float(accrued), day_index, params, rng, n_paths
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


def ridge_solve(design, target):
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def logit(x):
    x = np.clip(x, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(x / (1.0 - x))


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


def fit_default_proxy(accrued, values, day_index, params):
    exact = exact_tail_value(accrued, day_index, params)
    train_mask = ~np.isfinite(exact)
    if not np.any(train_mask):
        return lambda new_accrued: exact_tail_value(new_accrued, day_index, params)

    lower, upper = value_bounds(day_index, params)
    feature = expected_total_z(accrued[train_mask], day_index, params)
    scaled_feature, feature_low, feature_high = scale_to_unit(feature)
    normalized = (values[train_mask] - lower) / max(upper - lower, 1e-12)
    coeffs = ridge_solve(chebvander(scaled_feature, DEGREE), logit(normalized))

    def predict(new_accrued):
        new_accrued = np.asarray(new_accrued, dtype=float)
        exact_new = exact_tail_value(new_accrued, day_index, params)
        output = np.empty_like(new_accrued, dtype=float)
        exact_mask = np.isfinite(exact_new)
        output[exact_mask] = exact_new[exact_mask]
        fitted = ~exact_mask
        feature_new = expected_total_z(new_accrued[fitted], day_index, params)
        design = chebvander(apply_scale(feature_new, feature_low, feature_high), DEGREE)
        output[fitted] = lower + (upper - lower) * inv_logit(design @ coeffs)
        return np.clip(output, lower, upper)

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
    validation_grid = make_boundary_accrued_grid(day_index, params, VALIDATION_POINTS)
    benchmark, benchmark_stderr = build_labels(
        validation_grid, day_index, params, rng, BENCHMARK_PATHS_PER_STATE
    )
    train_grid = make_boundary_accrued_grid(day_index, params, TRAIN_POINTS)
    train_paths = paths_per_state_from_budget(TRAIN_SCENARIOS_PER_FIT, len(train_grid))
    train_values, train_stderr = build_labels(
        train_grid, day_index, params, rng, train_paths
    )
    proxy = fit_default_proxy(train_grid, train_values, day_index, params)
    prediction = proxy(validation_grid)

    return {
        "day_index": day_index,
        "time": day_index * period_dt(params),
        "remaining_periods": remaining_periods(day_index, params),
        "method": METHOD_NAME,
        "detail": METHOD_DETAIL,
        "train_states": len(train_grid),
        "train_scenarios_per_fit_target": TRAIN_SCENARIOS_PER_FIT,
        "train_scenarios_used": train_paths * len(train_grid),
        "train_paths_per_state": train_paths,
        "validation_states": len(validation_grid),
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
    rng = np.random.default_rng(params.seed + 719)
    rows = [run_day_index(params, day_index, rng) for day_index in TEST_DAY_INDICES]
    write_results(rows)

    print("Cliquet default proxy test")
    print(f"method: {METHOD_NAME} | {METHOD_DETAIL}")
    print(
        f"local floor/cap: {100 * params.local_floor:.1f}% / "
        f"{100 * params.local_cap:.1f}%"
    )
    print(
        f"global floor/cap: {100 * params.global_floor:.1f}% / "
        f"{100 * params.global_cap:.1f}%"
    )
    print(f"training scenarios/fit target: {TRAIN_SCENARIOS_PER_FIT:,}")
    print(f"benchmark paths/state: {BENCHMARK_PATHS_PER_STATE:,}")
    print()
    print("day    rem    max rel    p99 rel    MAE       max abs")
    print("---    ---    -------    -------    ---       -------")
    for row in rows:
        print(
            f"{row['day_index']:3d}    {row['remaining_periods']:3d}    "
            f"{100.0 * row['max_rel']:7.3f}%   "
            f"{100.0 * row['p99_rel']:7.3f}%   "
            f"{row['mae']:8.6f}  {row['max_abs']:8.6f}"
        )
    print()
    print(f"results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
