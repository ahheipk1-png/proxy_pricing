"""Standalone SLV cliquet proxy using the maturity-adaptive default."""

import csv
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.stats import norm, qmc


@dataclass(frozen=True)
class Params:
    s0: float = 100.0
    notional: float = 100.0
    rate: float = 0.05
    div_yield: float = 0.02
    maturity: float = 1.0
    n_periods: int = 12
    local_floor: float = -0.02
    local_cap: float = 0.04
    global_floor: float = 0.0
    global_cap: float = 0.20
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    vol_of_var: float = 0.35
    rho: float = -0.65
    local_skew: float = -0.25
    local_scale: float = 0.35
    seed: int = 43


TEST_MONTHS = [0, 3, 6, 9, 12]
TRAIN_STATES = 321
VALIDATION_STATES = 41
TRAIN_SCENARIOS_PER_FIT = 10_000_000
BENCHMARK_PATHS_PER_STATE = 524_288
STEPS_PER_PERIOD = 2
BATCH_PATHS = 131_072
SPOT_RANGE = (65.0, 150.0)
VAR_RANGE = (0.01, 0.12)
RELATIVE_ERROR_FLOOR = 0.01
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "SLVCliquetOptExperiment"
    / "default_run"
    / "slv_cliquet_default_results.csv"
)


def radical_inverse(index, base):
    value, fraction = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * fraction
        fraction /= base
    return value


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


def remaining(month, params):
    return params.n_periods - month


def period_dt(params):
    return params.maturity / params.n_periods


def discount(month, params):
    return exp(-params.rate * remaining(month, params) * period_dt(params))


def leverage(spot, params):
    x = np.log(np.maximum(spot, 1e-12) / params.s0)
    return np.clip(
        1.0 + params.local_skew * np.tanh(x / params.local_scale), 0.55, 1.45
    )


def payoff(total_return, params):
    return params.notional * np.clip(
        total_return, params.global_floor, params.global_cap
    )


def exact_tail(accrued, month, params):
    accrued = np.asarray(accrued)
    m = remaining(month, params)
    if m == 0:
        return payoff(accrued, params)
    result = np.full_like(accrued, np.nan, dtype=float)
    result[accrued + m * params.local_floor >= params.global_cap] = (
        discount(month, params) * params.notional * params.global_cap
    )
    result[accrued + m * params.local_cap <= params.global_floor] = (
        discount(month, params) * params.notional * params.global_floor
    )
    return result


def inverse_erf(x):
    x = np.asarray(x)
    a = 0.147
    logged = np.log(1.0 - x * x)
    term = 2.0 / (np.pi * a) + logged / 2.0
    return np.sign(x) * np.sqrt(np.sqrt(term * term - logged / a) - term)


def frozen_moments(spot, variance, params):
    count = 1001
    u = (np.arange(count) + 0.5) / count
    z = sqrt(2.0) * inverse_erf(2.0 * u - 1.0)
    sigma = leverage(np.asarray(spot), params) * np.sqrt(
        np.maximum(variance, 1e-10)
    )
    dt = period_dt(params)
    drift = (params.rate - params.div_yield - 0.5 * sigma**2) * dt
    returns = np.exp(drift[..., None] + sigma[..., None] * sqrt(dt) * z) - 1.0
    coupons = np.clip(returns, params.local_floor, params.local_cap)
    return np.mean(coupons, axis=-1), np.var(coupons, axis=-1)


def features(states, month, params):
    accrued, spot, variance = states
    m = remaining(month, params)
    if m == 0:
        zeros = np.zeros_like(accrued)
        return zeros, zeros, zeros
    mean, var = frozen_moments(spot, variance, params)
    midpoint = 0.5 * (params.global_floor + params.global_cap)
    z = (accrued + m * mean - midpoint) / np.sqrt(np.maximum(m * var, 1e-8))
    return (
        z,
        np.log(np.maximum(spot, 1e-12) / params.s0),
        np.log(np.maximum(variance, 1e-8) / params.theta),
    )


def make_states(month, params, count, validation=False):
    low = month * params.local_floor if month else 0.0
    high = month * params.local_cap if month else 0.0
    accrued, spot, variance = (np.empty(count) for _ in range(3))
    offset = 5003 if validation else 0
    for row in range(count):
        index = row + 1 + offset
        ua = radical_inverse(index, 2)
        us = radical_inverse(index, 3)
        uv = radical_inverse(index, 5)
        spot[row] = SPOT_RANGE[0] * (SPOT_RANGE[1] / SPOT_RANGE[0]) ** us
        variance[row] = VAR_RANGE[0] * (VAR_RANGE[1] / VAR_RANGE[0]) ** uv
        if high <= low:
            accrued[row] = low
        elif row % 4 == 0:
            accrued[row] = low + ua * (high - low)
        else:
            mean, var = frozen_moments(
                np.array([spot[row]]), np.array([variance[row]]), params
            )
            boundary = (
                params.global_floor if row % 4 in {1, 2} else params.global_cap
            )
            center = boundary - remaining(month, params) * mean[0]
            width = max(
                sqrt(max(remaining(month, params) * var[0], 1e-8)),
                (high - low) / 40.0,
            )
            accrued[row] = np.clip(
                center + (2.0 * ua - 1.0) * 5.0 * width, low, high
            )
    return accrued, spot, variance


def simulate(accrued, spot0, variance0, month, params, rng, n_paths):
    exact = exact_tail(np.array([accrued]), month, params)[0]
    if np.isfinite(exact):
        return float(exact), 0.0
    steps = remaining(month, params) * STEPS_PER_PERIOD
    dt = period_dt(params) / STEPS_PER_PERIOD
    rho_scale = sqrt(1.0 - params.rho**2)
    coupon_mean, _ = frozen_moments(
        np.array([spot0]), np.array([variance0]), params
    )
    expected_total = accrued + remaining(month, params) * coupon_mean[0]
    required_coupon = max(
        (params.global_floor - expected_total) / max(remaining(month, params), 1),
        0.0,
    )
    independent_month_vol = (
        leverage(np.array([spot0]), params)[0]
        * sqrt(max(variance0 * period_dt(params), 1e-10))
        * rho_scale
    )
    importance_shift = (
        np.clip(
            required_coupon
            / max(independent_month_vol * sqrt(STEPS_PER_PERIOD), 1e-8)
            + 0.20,
            0.0,
            0.75,
        )
        if required_coupon > 0.0
        else 0.0
    )
    total = total_sq = 0.0
    count = 0
    seed = rng.integers(0, np.iinfo(np.int64).max)
    for base_normals in sobol_antithetic_batches(
        n_paths, 2 * steps, seed, BATCH_PATHS
    ):
        batch = len(base_normals)
        z1 = base_normals[:, :steps]
        z2 = importance_shift + base_normals[:, steps:]
        likelihood = np.exp(
            -importance_shift * np.sum(z2, axis=1)
            + 0.5 * steps * importance_shift**2
        )
        spot = np.full(batch, spot0)
        variance = np.full(batch, variance0)
        reset_spot = spot.copy()
        future_sum = np.zeros(batch)
        for step in range(steps):
            positive_var = np.maximum(variance, 0.0)
            root_var = np.sqrt(positive_var)
            z_var = z1[:, step]
            z_spot = params.rho * z_var + rho_scale * z2[:, step]
            lev = leverage(spot, params)
            inst_var = lev * lev * positive_var
            spot *= np.exp(
                (params.rate - params.div_yield - 0.5 * inst_var) * dt
                + lev * root_var * sqrt(dt) * z_spot
            )
            variance = np.maximum(
                variance
                + params.kappa * (params.theta - positive_var) * dt
                + params.vol_of_var * root_var * sqrt(dt) * z_var,
                0.0,
            )
            if (step + 1) % STEPS_PER_PERIOD == 0:
                future_sum += np.clip(
                    spot / reset_spot - 1.0, params.local_floor, params.local_cap
                )
                reset_spot = spot.copy()
        value = (
            discount(month, params)
            * payoff(accrued + future_sum, params)
            * likelihood
        )
        count += batch
        total += float(np.sum(value))
        total_sq += float(np.sum(value * value))
    mean = total / count
    sample_var = max((total_sq - count * mean * mean) / max(count - 1, 1), 0.0)
    return mean, sqrt(sample_var / count)


def labels(states, month, params, rng, paths):
    value, stderr = np.empty(len(states[0])), np.empty(len(states[0]))
    common_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    for row in range(len(value)):
        value[row], stderr[row] = simulate(
            states[0][row],
            states[1][row],
            states[2][row],
            month,
            params,
            np.random.default_rng(common_seed),
            paths,
        )
    return value, stderr


def scaled(values):
    low, high = float(np.min(values)), float(np.max(values))
    if high <= low + 1e-12:
        return np.zeros_like(values), low, high
    return 2.0 * (values - low) / (high - low) - 1.0, low, high


def use_scale(values, low, high):
    if high <= low + 1e-12:
        return np.zeros_like(values)
    return np.clip(2.0 * (values - low) / (high - low) - 1.0, -1.0, 1.0)


def anisotropic_design(z, spot, variance):
    tz, ts, tv = chebvander(z, 19), chebvander(spot, 3), chebvander(variance, 3)
    columns = []
    for i in range(20):
        for j in range(4):
            for k in range(4):
                if i + 5 * j + 5 * k <= 19:
                    columns.append(tz[:, i] * ts[:, j] * tv[:, k])
    return np.column_stack(columns)


def fit_default(states, values, month, params):
    exact = exact_tail(states[0], month, params)
    active = ~np.isfinite(exact)
    if not np.any(active):
        return lambda new_states: exact_tail(new_states[0], month, params)
    lower = discount(month, params) * params.notional * params.global_floor
    upper = discount(month, params) * params.notional * params.global_cap
    normalized = np.clip(
        (values[active] - lower) / max(upper - lower, 1e-12), 1e-7, 1.0 - 1e-7
    )
    target = np.log(normalized / (1.0 - normalized))
    z, s, v = features(tuple(x[active] for x in states), month, params)
    zs, z_low, z_high = scaled(z)
    ss, s_low, s_high = scaled(s)
    vs, v_low, v_high = scaled(v)
    train_feature = np.column_stack((zs, ss, vs))

    if remaining(month, params) < 9:
        design = anisotropic_design(zs, ss, vs)
        penalty = np.eye(design.shape[1]) * 3e-7
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ target
        )

    def transformed(new_states):
        new_z, new_s, new_v = features(new_states, month, params)
        return np.column_stack(
            (
                use_scale(new_z, z_low, z_high),
                use_scale(new_s, s_low, s_high),
                use_scale(new_v, v_low, v_high),
            )
        )

    def raw_predict(new_states):
        query = transformed(new_states)
        if remaining(month, params) < 9:
            return anisotropic_design(*query.T) @ coefficients
        fitted = np.empty(len(query))
        neighbor_count = min(64, len(train_feature))
        for row, point in enumerate(query):
            delta = train_feature - point
            distance_sq = np.sum(delta * delta, axis=1)
            neighbors = np.argpartition(
                distance_sq, neighbor_count - 1
            )[:neighbor_count]
            local = delta[neighbors]
            bandwidth = max(float(np.max(distance_sq[neighbors])), 1e-8)
            weights = np.exp(-3.0 * distance_sq[neighbors] / bandwidth)
            dz, ds, dv = local.T
            design = np.column_stack(
                (
                    np.ones(neighbor_count),
                    dz,
                    ds,
                    dv,
                    dz * dz,
                    ds * ds,
                    dv * dv,
                    dz * ds,
                    dz * dv,
                    ds * dv,
                )
            )
            root_weight = np.sqrt(weights)
            weighted = design * root_weight[:, None]
            penalty = np.eye(design.shape[1]) * 1e-5
            penalty[0, 0] = 0.0
            coefficient = np.linalg.solve(
                weighted.T @ weighted + penalty,
                weighted.T @ (target[neighbors] * root_weight),
            )
            fitted[row] = coefficient[0]
        return fitted

    def predict(new_states):
        exact_new = exact_tail(new_states[0], month, params)
        output = np.empty_like(new_states[0])
        exact_mask = np.isfinite(exact_new)
        output[exact_mask] = exact_new[exact_mask]
        if np.any(~exact_mask):
            subset = tuple(x[~exact_mask] for x in new_states)
            raw = raw_predict(subset)
            bounded = 1.0 / (1.0 + np.exp(-np.clip(raw, -35.0, 35.0)))
            output[~exact_mask] = lower + (upper - lower) * bounded
        return np.clip(output, lower, upper)

    return predict


def main():
    params = Params()
    rng = np.random.default_rng(params.seed)
    rows = []
    train_paths = qmc_path_count(np.ceil(TRAIN_SCENARIOS_PER_FIT / TRAIN_STATES))
    for month in TEST_MONTHS:
        train_states = make_states(month, params, TRAIN_STATES)
        test_states = make_states(month, params, VALIDATION_STATES, validation=True)
        train_value, _ = labels(
            train_states, month, params, rng, train_paths
        )
        benchmark, stderr = labels(
            test_states, month, params, rng, BENCHMARK_PATHS_PER_STATE
        )
        proxy = fit_default(train_states, train_value, month, params)
        prediction = proxy(test_states)
        absolute = np.abs(prediction - benchmark)
        relative = absolute / np.maximum(np.abs(benchmark), RELATIVE_ERROR_FLOOR)
        rows.append(
            {
                "month": month,
                "remaining_periods": remaining(month, params),
                "component": (
                    "local_quadratic"
                    if remaining(month, params) >= 9
                    else "anisotropic_chebyshev_d19"
                ),
                "max_rel": float(np.max(relative)),
                "p99_rel": float(np.quantile(relative, 0.99)),
                "mae": float(np.mean(absolute)),
                "max_abs": float(np.max(absolute)),
                "avg_benchmark_stderr": float(np.mean(stderr)),
            }
        )
        print(f"finished reset month {month}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSLV cliquet default proxy test")
    print("method: maturity-adaptive 3D bounded proxy")
    print("month   component                  max rel    p99 rel    MAE")
    for row in rows:
        print(
            f"{row['month']:5d}   {row['component']:<25} "
            f"{100 * row['max_rel']:7.3f}%   {100 * row['p99_rel']:7.3f}%   "
            f"{row['mae']:8.6f}"
        )
    print(f"\nresults written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
