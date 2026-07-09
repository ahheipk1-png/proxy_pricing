import csv
from dataclasses import dataclass
from math import exp, sqrt
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
class Params:
    s0: tuple = (100.0, 100.0, 100.0)
    notional: float = 100.0
    rate: float = 0.05
    maturity: float = 1.0
    n_periods: int = 12
    local_floor: float = -0.02
    local_cap: float = 0.04
    global_floor: float = 0.0
    global_cap: float = 0.20
    div_yields: tuple = (0.020, 0.015, 0.025)
    v0: tuple = (0.0324, 0.0484, 0.0784)
    kappa: tuple = (2.0, 1.8, 1.5)
    theta: tuple = (0.0324, 0.0484, 0.0784)
    vol_of_var: tuple = (0.30, 0.38, 0.45)
    spot_var_rho: tuple = (-0.55, -0.65, -0.70)
    local_skew: tuple = (-0.20, -0.25, -0.30)
    local_scale: tuple = (0.35, 0.35, 0.35)
    market_correlation: tuple = (
        (1.00, 0.75, 0.55),
        (0.75, 1.00, 0.65),
        (0.55, 0.65, 1.00),
    )
    seed: int = 83


VARIANTS = {
    "basket_return": "Clip the equal-weight basket return",
    "average_clipped": "Average the three individually clipped returns",
    "worst_of": "Clip the worst-performing underlying return",
}
METHODS = [
    "local_summary_quadratic",
    "local_full_quadratic",
    "sparse_chebyshev",
    "adaptive_blend",
]
TEST_MONTHS = [0, 3, 6, 9, 12]
TRAIN_STATES = 2001
VALIDATION_STATES = 31
TRAIN_SCENARIOS_PER_FIT = 10_000_000
BENCHMARK_PATHS_PER_STATE = 524_288
STEPS_PER_PERIOD = 2
SIMULATION_BATCH = 131_072
RELATIVE_ERROR_FLOOR = 0.01
SPOT_RANGE = (65.0, 150.0)
VAR_MULTIPLIER_RANGE = (0.35, 3.0)
LOGIT_EPS = 1e-7

OUTPUT_DIR = Path(__file__).resolve().parent / "BasketCliquetOptExperiment" / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
METHOD_CSV = OUTPUT_DIR / "basket_slv_cliquet_proxy_method_results.csv"
DETAIL_CSV = OUTPUT_DIR / "basket_slv_cliquet_proxy_validation_details.csv"
TRAINING_CSV = OUTPUT_DIR / "basket_slv_cliquet_training_labels.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"


def remaining_periods(month, params):
    return params.n_periods - month


def power_of_two_at_least(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 1))))


def qmc_path_count(target_paths):
    return 4 * power_of_two_at_least((int(target_paths) + 3) // 4)


def sobol_normals(n_points, dimension, seed):
    count = power_of_two_at_least(n_points)
    engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    uniforms = engine.random_base2(int(np.log2(count)))
    uniforms = np.clip(uniforms, 1e-12, 1.0 - 1e-12)
    return norm.ppf(uniforms)


def sobol_quarter_batches(target_paths, dimension, seed, max_batch):
    quarter_total = power_of_two_at_least((int(target_paths) + 3) // 4)
    quarter_batch = power_of_two_at_least(max(1, int(max_batch) // 4))
    quarter_batch = min(quarter_batch, quarter_total)
    engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    remaining = quarter_total
    while remaining:
        quarter = min(quarter_batch, remaining)
        uniforms = engine.random(quarter)
        uniforms = np.clip(uniforms, 1e-12, 1.0 - 1e-12)
        yield norm.ppf(uniforms)
        remaining -= quarter


def period_dt(params):
    return params.maturity / params.n_periods


def discount(month, params):
    return exp(-params.rate * remaining_periods(month, params) * period_dt(params))


def payoff(total_return, params):
    return params.notional * np.clip(
        total_return, params.global_floor, params.global_cap
    )


def accrued_range(month, params):
    if month == 0:
        return 0.0, 0.0
    return month * params.local_floor, month * params.local_cap


def exact_tail(accrued, month, params):
    accrued = np.asarray(accrued, dtype=float)
    m = remaining_periods(month, params)
    if m == 0:
        return payoff(accrued, params)
    result = np.full_like(accrued, np.nan)
    result[accrued + m * params.local_floor >= params.global_cap] = (
        discount(month, params) * params.notional * params.global_cap
    )
    result[accrued + m * params.local_cap <= params.global_floor] = (
        discount(month, params) * params.notional * params.global_floor
    )
    return result


def leverage(spots, params):
    s0 = np.asarray(params.s0)
    skew = np.asarray(params.local_skew)
    scale = np.asarray(params.local_scale)
    x = np.log(np.maximum(spots, 1e-12) / s0)
    return np.clip(1.0 + skew * np.tanh(x / scale), 0.50, 1.50)


def coupon_values(returns, params):
    return {
        "basket_return": np.clip(
            np.mean(returns, axis=-1), params.local_floor, params.local_cap
        ),
        "average_clipped": np.mean(
            np.clip(returns, params.local_floor, params.local_cap), axis=-1
        ),
        "worst_of": np.clip(
            np.min(returns, axis=-1), params.local_floor, params.local_cap
        ),
    }


def feature_normals(params, n_paths=4096):
    half = power_of_two_at_least(n_paths // 2)
    base = sobol_normals(half, 6, params.seed + 1009)
    z_var = np.vstack((base[:, :3], -base[:, :3]))
    z_market = np.vstack((base[:, 3:], -base[:, 3:]))
    market_chol = np.linalg.cholesky(np.asarray(params.market_correlation))
    market = z_market @ market_chol.T
    rho = np.asarray(params.spot_var_rho)
    spot_z = rho * z_var + np.sqrt(1.0 - rho * rho) * market
    return spot_z


def frozen_coupon_moments(spots, variances, params, spot_z):
    spots = np.asarray(spots)
    variances = np.asarray(variances)
    dt = period_dt(params)
    divs = np.asarray(params.div_yields)
    lev = leverage(spots, params)
    instantaneous_var = lev * lev * np.maximum(variances, 1e-10)
    drift = (params.rate - divs - 0.5 * instantaneous_var) * dt
    diffusion = lev * np.sqrt(np.maximum(variances, 1e-10) * dt)
    returns = np.exp(drift + diffusion * spot_z) - 1.0
    coupons = coupon_values(returns, params)
    output = {}
    for name, values in coupons.items():
        mean = float(np.mean(values))
        variance = float(np.var(values))
        stdev = sqrt(max(variance, 1e-14))
        skewness = float(np.mean(((values - mean) / stdev) ** 3))
        floor_mass = float(np.mean(values <= params.local_floor + 1e-12))
        cap_mass = float(np.mean(values >= params.local_cap - 1e-12))
        output[name] = (mean, variance, skewness, floor_mass, cap_mass)
    return output


def radical_inverse(index, base):
    value, fraction = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * fraction
        fraction /= base
    return value


def make_states(
    month, params, count, spot_z, validation=False, validation_offset=None
):
    primes = (2, 3, 5, 7, 11, 13, 17)
    offset = (
        (5003 if validation_offset is None else validation_offset)
        if validation
        else 0
    )
    low, high = accrued_range(month, params)
    theta = np.asarray(params.theta)
    accrued = np.empty(count)
    spots = np.empty((count, 3))
    variances = np.empty((count, 3))
    feature_moments = []

    for row in range(count):
        index = row + 1 + offset
        u = [radical_inverse(index, base) for base in primes]
        for asset in range(3):
            spots[row, asset] = SPOT_RANGE[0] * (
                SPOT_RANGE[1] / SPOT_RANGE[0]
            ) ** u[1 + asset]
            variances[row, asset] = theta[asset] * (
                VAR_MULTIPLIER_RANGE[0]
                * (
                    VAR_MULTIPLIER_RANGE[1] / VAR_MULTIPLIER_RANGE[0]
                )
                ** u[4 + asset]
            )
        moments = frozen_coupon_moments(
            spots[row], variances[row], params, spot_z
        )
        feature_moments.append(moments)
        if high <= low:
            accrued[row] = low
        elif row % 4 == 0:
            accrued[row] = low + u[0] * (high - low)
        else:
            variant = tuple(VARIANTS)[row % len(VARIANTS)]
            mean, variance = moments[variant][:2]
            boundary = params.global_floor if row % 4 in {1, 2} else params.global_cap
            m = remaining_periods(month, params)
            center = boundary - m * mean
            width = max(sqrt(max(m * variance, 1e-8)), (high - low) / 40.0)
            accrued[row] = np.clip(
                center + (2.0 * u[0] - 1.0) * 5.0 * width, low, high
            )
    return (accrued, spots, variances), feature_moments


def mixture_shift(accrued, month, moments, spots, variances, params):
    m = remaining_periods(month, params)
    required = 0.0
    for variant in VARIANTS:
        expected_total = accrued + m * moments[variant][0]
        required = max(
            required,
            (params.global_floor - expected_total) / max(m, 1),
        )
    if required <= 0.0:
        return np.zeros(3)
    effective_month_vol = float(
        np.mean(
            leverage(spots, params)
            * np.sqrt(np.maximum(variances, 1e-10) * period_dt(params))
        )
    )
    common_shift = np.clip(
        required / max(effective_month_vol * sqrt(STEPS_PER_PERIOD), 1e-8)
        + 0.15,
        0.0,
        0.65,
    )
    market_chol = np.linalg.cholesky(np.asarray(params.market_correlation))
    direction = np.linalg.solve(market_chol, np.ones(3))
    return common_shift * direction


def simulate_state(
    accrued,
    spots0,
    variances0,
    month,
    moments,
    params,
    rng,
    n_paths,
):
    exact = exact_tail(np.array([accrued]), month, params)[0]
    if np.isfinite(exact):
        values = {name: float(exact) for name in VARIANTS}
        stderrs = {name: 0.0 for name in VARIANTS}
        return values, stderrs

    m = remaining_periods(month, params)
    n_steps = m * STEPS_PER_PERIOD
    dt = period_dt(params) / STEPS_PER_PERIOD
    divs = np.asarray(params.div_yields)
    kappa = np.asarray(params.kappa)
    theta = np.asarray(params.theta)
    vol_of_var = np.asarray(params.vol_of_var)
    rho = np.asarray(params.spot_var_rho)
    rho_scale = np.sqrt(1.0 - rho * rho)
    market_chol = np.linalg.cholesky(np.asarray(params.market_correlation))
    shift = mixture_shift(
        accrued, month, moments, spots0, variances0, params
    )
    shift_norm_sq = float(shift @ shift)
    totals = {name: 0.0 for name in VARIANTS}
    totals_sq = {name: 0.0 for name in VARIANTS}
    count = 0
    seed = rng.integers(0, np.iinfo(np.int64).max)
    qmc_dimension = 2 * n_steps * 3
    for base in sobol_quarter_batches(n_paths, qmc_dimension, seed, SIMULATION_BATCH):
        quarter = len(base)
        batch = 4 * quarter
        base_var = base[:, : n_steps * 3].reshape(quarter, n_steps, 3)
        base_market = base[:, n_steps * 3 :].reshape(quarter, n_steps, 3)
        z_var = np.concatenate(
            (base_var, -base_var, base_var, -base_var), axis=0
        )
        z_market = np.concatenate(
            (
                base_market,
                -base_market,
                base_market + shift,
                -base_market + shift,
            ),
            axis=0,
        )
        log_ratio = (
            np.sum(z_market @ shift, axis=1)
            - 0.5 * n_steps * shift_norm_sq
        )
        likelihood = 1.0 / (
            0.5 + 0.5 * np.exp(np.clip(log_ratio, -50.0, 50.0))
        )
        market = z_market @ market_chol.T
        spots = np.broadcast_to(spots0, (batch, 3)).copy()
        variances = np.broadcast_to(variances0, (batch, 3)).copy()
        reset_spots = spots.copy()
        sums = {name: np.zeros(batch) for name in VARIANTS}

        for step in range(n_steps):
            positive_var = np.maximum(variances, 0.0)
            root_var = np.sqrt(positive_var)
            spot_z = rho * z_var[:, step, :] + rho_scale * market[:, step, :]
            lev = leverage(spots, params)
            instantaneous_var = lev * lev * positive_var
            spots *= np.exp(
                (params.rate - divs - 0.5 * instantaneous_var) * dt
                + lev * root_var * sqrt(dt) * spot_z
            )
            variances = np.maximum(
                variances
                + kappa * (theta - positive_var) * dt
                + vol_of_var * root_var * sqrt(dt) * z_var[:, step, :],
                0.0,
            )
            if (step + 1) % STEPS_PER_PERIOD == 0:
                period_returns = spots / reset_spots - 1.0
                coupons = coupon_values(period_returns, params)
                for variant in VARIANTS:
                    sums[variant] += coupons[variant]
                reset_spots = spots.copy()

        df = discount(month, params)
        for variant in VARIANTS:
            sample = df * payoff(accrued + sums[variant], params) * likelihood
            totals[variant] += float(np.sum(sample))
            totals_sq[variant] += float(np.sum(sample * sample))
        count += batch

    values = {}
    stderrs = {}
    for variant in VARIANTS:
        mean = totals[variant] / count
        variance = max(
            (totals_sq[variant] - count * mean * mean) / max(count - 1, 1),
            0.0,
        )
        values[variant] = mean
        stderrs[variant] = sqrt(variance / count)
    return values, stderrs


def build_labels(states, feature_moments, month, params, rng, paths):
    count = len(states[0])
    values = {name: np.empty(count) for name in VARIANTS}
    stderrs = {name: np.empty(count) for name in VARIANTS}
    common_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    for row in range(count):
        state_values, state_stderrs = simulate_state(
            states[0][row],
            states[1][row],
            states[2][row],
            month,
            feature_moments[row],
            params,
            np.random.default_rng(common_seed),
            paths,
        )
        for variant in VARIANTS:
            values[variant][row] = state_values[variant]
            stderrs[variant][row] = state_stderrs[variant]
    return values, stderrs


def raw_features(states, moments, month, variant, params, summary):
    accrued, spots, variances = states
    m = remaining_periods(month, params)
    means = np.array([item[variant][0] for item in moments])
    variances_coupon = np.array([item[variant][1] for item in moments])
    expected_total = accrued + m * means
    stdev = np.sqrt(np.maximum(m * variances_coupon, 1e-8))
    lower_cushion = (expected_total - params.global_floor) / stdev
    upper_cushion = (params.global_cap - expected_total) / stdev
    log_spots = np.log(spots / np.asarray(params.s0))
    log_vars = np.log(
        np.maximum(variances, 1e-10) / np.asarray(params.theta)
    )
    if not summary:
        return np.column_stack(
            (
                lower_cushion,
                upper_cushion,
                log_spots,
                log_vars,
            )
        )
    return np.column_stack(
        (
            lower_cushion,
            upper_cushion,
            np.mean(log_spots, axis=1),
            np.std(log_spots, axis=1),
            np.mean(log_vars, axis=1),
            np.std(log_vars, axis=1),
        )
    )


def neural_features(states, moments, month, variant, params):
    accrued, spots, variances = states
    m = remaining_periods(month, params)
    mean = np.array([item[variant][0] for item in moments])
    coupon_var = np.array([item[variant][1] for item in moments])
    skewness = np.array([item[variant][2] for item in moments])
    floor_mass = np.array([item[variant][3] for item in moments])
    cap_mass = np.array([item[variant][4] for item in moments])
    expected_total = accrued + m * mean
    stdev = np.sqrt(np.maximum(m * coupon_var, 1e-8))
    log_spots = np.log(spots / np.asarray(params.s0))
    log_vars = np.log(
        np.maximum(variances, 1e-10) / np.asarray(params.theta)
    )
    return np.column_stack(
        (
            (expected_total - params.global_floor) / stdev,
            (params.global_cap - expected_total) / stdev,
            skewness,
            floor_mass,
            cap_mass,
            log_spots,
            log_vars,
        )
    )


def scale_features(features):
    low = np.min(features, axis=0)
    high = np.max(features, axis=0)
    width = np.where(high > low + 1e-12, high - low, 1.0)
    return 2.0 * (features - low) / width - 1.0, low, high


def apply_feature_scale(features, low, high):
    width = np.where(high > low + 1e-12, high - low, 1.0)
    return np.clip(2.0 * (features - low) / width - 1.0, -1.0, 1.0)


def polynomial_design(delta, quadratic):
    columns = [np.ones(len(delta))]
    columns.extend(delta[:, index] for index in range(delta.shape[1]))
    if quadratic:
        for left in range(delta.shape[1]):
            for right in range(left, delta.shape[1]):
                columns.append(delta[:, left] * delta[:, right])
    return np.column_stack(columns)


def local_predict(train_feature, train_target, query, quadratic):
    dimension = train_feature.shape[1]
    term_count = 1 + dimension + (
        dimension * (dimension + 1) // 2 if quadratic else 0
    )
    neighbor_count = min(max(3 * term_count, 64), len(train_feature))
    output = np.empty(len(query))
    for row, point in enumerate(query):
        delta = train_feature - point
        distance_sq = np.sum(delta * delta, axis=1)
        neighbors = np.argpartition(
            distance_sq, neighbor_count - 1
        )[:neighbor_count]
        local_delta = delta[neighbors]
        bandwidth = max(float(np.max(distance_sq[neighbors])), 1e-8)
        weights = np.exp(-3.0 * distance_sq[neighbors] / bandwidth)
        design = polynomial_design(local_delta, quadratic)
        root_weight = np.sqrt(weights)
        weighted_design = design * root_weight[:, None]
        penalty = np.eye(design.shape[1]) * (3e-5 if quadratic else 1e-6)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            weighted_design.T @ weighted_design + penalty,
            weighted_design.T @ (train_target[neighbors] * root_weight),
        )
        output[row] = coefficients[0]
    return output


def sparse_chebyshev_design(features):
    lower = features[:, 0]
    upper = features[:, 1]
    state = features[:, 2:]
    t_lower = chebvander(lower, 15)
    t_upper = chebvander(upper, 8)
    columns = [t_lower[:, degree] for degree in range(16)]
    columns.extend(t_upper[:, degree] for degree in range(1, 9))
    for lower_degree in range(1, 9):
        for upper_degree in range(1, 9 - lower_degree):
            columns.append(
                t_lower[:, lower_degree] * t_upper[:, upper_degree]
            )
    for dimension in range(state.shape[1]):
        for degree in range(6):
            columns.append(t_lower[:, degree] * state[:, dimension])
        for degree in range(4):
            columns.append(t_upper[:, degree] * state[:, dimension])
    for left in range(state.shape[1]):
        for right in range(left + 1, state.shape[1]):
            columns.append(state[:, left] * state[:, right])
    return np.column_stack(columns)


def rbf_kernel(left, right, length_scale):
    left_sq = np.sum(left * left, axis=1)[:, None]
    right_sq = np.sum(right * right, axis=1)[None, :]
    distance_sq = np.maximum(
        left_sq + right_sq - 2.0 * left @ right.T, 0.0
    )
    return np.exp(-0.5 * distance_sq / (length_scale * length_scale))


def fit_tanh_mlp(train_x, train_y, seed, hidden=48, epochs=2500):
    x_mean = np.mean(train_x, axis=0)
    x_std = np.maximum(np.std(train_x, axis=0), 1e-8)
    x = (train_x - x_mean) / x_std
    y_mean = float(np.mean(train_y))
    y_std = max(float(np.std(train_y)), 1e-8)
    y = ((train_y - y_mean) / y_std)[:, None]
    rng = np.random.default_rng(seed)
    parameters = [
        rng.standard_normal((x.shape[1], hidden)) / sqrt(x.shape[1]),
        np.zeros((1, hidden)),
        rng.standard_normal((hidden, hidden)) / sqrt(hidden),
        np.zeros((1, hidden)),
        rng.standard_normal((hidden, 1)) / sqrt(hidden),
        np.zeros((1, 1)),
    ]
    first_moment = [np.zeros_like(item) for item in parameters]
    second_moment = [np.zeros_like(item) for item in parameters]
    beta1, beta2 = 0.9, 0.999

    for epoch in range(1, epochs + 1):
        w1, b1, w2, b2, w3, b3 = parameters
        h1 = np.tanh(x @ w1 + b1)
        h2 = np.tanh(h1 @ w2 + b2)
        prediction = h2 @ w3 + b3
        output_gradient = 2.0 * (prediction - y) / len(x)
        gradients = [None] * 6
        gradients[4] = h2.T @ output_gradient + 2e-5 * w3
        gradients[5] = np.sum(output_gradient, axis=0, keepdims=True)
        h2_gradient = (output_gradient @ w3.T) * (1.0 - h2 * h2)
        gradients[2] = h1.T @ h2_gradient + 2e-5 * w2
        gradients[3] = np.sum(h2_gradient, axis=0, keepdims=True)
        h1_gradient = (h2_gradient @ w2.T) * (1.0 - h1 * h1)
        gradients[0] = x.T @ h1_gradient + 2e-5 * w1
        gradients[1] = np.sum(h1_gradient, axis=0, keepdims=True)
        learning_rate = 0.004 * (0.25 + 0.75 * (1.0 - epoch / epochs))
        for index in range(6):
            first_moment[index] = (
                beta1 * first_moment[index] + (1.0 - beta1) * gradients[index]
            )
            second_moment[index] = (
                beta2 * second_moment[index]
                + (1.0 - beta2) * gradients[index] ** 2
            )
            m_hat = first_moment[index] / (1.0 - beta1**epoch)
            v_hat = second_moment[index] / (1.0 - beta2**epoch)
            parameters[index] -= (
                learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            )

    def predict(new_x):
        scaled_x = (new_x - x_mean) / x_std
        w1, b1, w2, b2, w3, b3 = parameters
        output = np.tanh(
            np.tanh(scaled_x @ w1 + b1) @ w2 + b2
        ) @ w3 + b3
        return output[:, 0] * y_std + y_mean

    return predict


def fit_proxy(
    states,
    moments,
    values,
    month,
    variant,
    params,
    method,
):
    if method == "adaptive_blend":
        local_method = (
            "local_full_quadratic"
            if variant == "worst_of"
            else "local_summary_quadratic"
        )
        local_proxy = fit_proxy(
            states, moments, values, month, variant, params, local_method
        )
        spectral_proxy = fit_proxy(
            states,
            moments,
            values,
            month,
            variant,
            params,
            "sparse_chebyshev",
        )
        local_weight = 0.16 + 0.02 * remaining_periods(month, params)

        def blended(new_states, new_moments):
            return (
                local_weight * local_proxy(new_states, new_moments)
                + (1.0 - local_weight)
                * spectral_proxy(new_states, new_moments)
            )

        return blended
    exact = exact_tail(states[0], month, params)
    active = ~np.isfinite(exact)
    if not np.any(active):
        return lambda new_states, new_moments: exact_tail(
            new_states[0], month, params
        )
    lower = discount(month, params) * params.notional * params.global_floor
    upper = discount(month, params) * params.notional * params.global_cap
    normalized = np.clip(
        (values[active] - lower) / max(upper - lower, 1e-12),
        LOGIT_EPS,
        1.0 - LOGIT_EPS,
    )
    target = np.log(normalized / (1.0 - normalized))
    summary = method in {"local_summary_quadratic", "rbf_summary"}
    full_features = (
        neural_features(states, moments, month, variant, params)
        if method == "mlp_tanh_ensemble"
        else raw_features(
            states, moments, month, variant, params, summary=summary
        )
    )
    train_feature, feature_low, feature_high = scale_features(
        full_features[active]
    )

    if method == "sparse_chebyshev":
        design = sparse_chebyshev_design(train_feature)
        penalty = np.eye(design.shape[1]) * 1e-5
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ target
        )
    elif method in {"rbf_summary", "rbf_full"}:
        pairwise = np.sqrt(
            np.maximum(
                np.sum(train_feature * train_feature, axis=1)[:, None]
                + np.sum(train_feature * train_feature, axis=1)[None, :]
                - 2.0 * train_feature @ train_feature.T,
                0.0,
            )
        )
        nonzero = pairwise[pairwise > 1e-12]
        length_scale = 0.55 * float(np.median(nonzero))
        kernel = rbf_kernel(train_feature, train_feature, length_scale)
        coefficients = np.linalg.solve(
            kernel + 3e-4 * np.eye(len(kernel)), target
        )
    elif method == "mlp_tanh_ensemble":
        variant_index = list(VARIANTS).index(variant)
        mlp_models = [
            fit_tanh_mlp(
                full_features[active],
                np.clip(target, -14.0, 14.0),
                seed=params.seed + 1000 * month + 100 * variant_index + member,
            )
            for member in range(3)
        ]

    def raw_prediction(new_states, new_moments):
        new_features = (
            neural_features(
                new_states, new_moments, month, variant, params
            )
            if method == "mlp_tanh_ensemble"
            else raw_features(
                new_states,
                new_moments,
                month,
                variant,
                params,
                summary=summary,
            )
        )
        if method == "mlp_tanh_ensemble":
            return np.mean(
                [model(new_features) for model in mlp_models], axis=0
            )
        query = apply_feature_scale(new_features, feature_low, feature_high)
        if method == "local_summary_quadratic":
            return local_predict(train_feature, target, query, quadratic=True)
        if method == "local_full_linear":
            return local_predict(train_feature, target, query, quadratic=False)
        if method == "local_full_quadratic":
            return local_predict(train_feature, target, query, quadratic=True)
        if method == "sparse_chebyshev":
            return sparse_chebyshev_design(query) @ coefficients
        if method in {"rbf_summary", "rbf_full"}:
            return rbf_kernel(query, train_feature, length_scale) @ coefficients
        raise ValueError(method)

    def predict(new_states, new_moments):
        exact_new = exact_tail(new_states[0], month, params)
        result = np.empty_like(new_states[0])
        tail = np.isfinite(exact_new)
        result[tail] = exact_new[tail]
        if np.any(~tail):
            subset_states = (
                new_states[0][~tail],
                new_states[1][~tail],
                new_states[2][~tail],
            )
            subset_moments = [
                moment for moment, use in zip(new_moments, ~tail) if use
            ]
            raw = raw_prediction(subset_states, subset_moments)
            bounded = 1.0 / (1.0 + np.exp(-np.clip(raw, -35.0, 35.0)))
            result[~tail] = lower + (upper - lower) * bounded
        return np.clip(result, lower, upper)

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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_plot(path, variant, month, benchmark, prediction):
    if Image is None:
        return
    width, height = 1200, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panels = [(60, 65, 1140, 350), (60, 390, 1140, 650), (60, 690, 1140, 940)]
    order = np.argsort(benchmark)
    x = np.arange(len(benchmark), dtype=float)
    error = prediction - benchmark
    relative = 100.0 * error / np.maximum(np.abs(benchmark), RELATIVE_ERROR_FLOOR)
    series = [
        [("benchmark", benchmark[order], (35, 90, 180)), ("proxy", prediction[order], (210, 55, 55))],
        [("signed error", error[order], (25, 145, 85))],
        [("signed relative error (%)", relative[order], (125, 70, 170))],
    ]
    draw.text(
        (60, 20),
        f"3-asset SLV {variant}, reset month {month}; states ranked by benchmark",
        fill="black",
    )
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
            for x_value, y_value in zip(x, values):
                px = left + x_value / max(len(x) - 1, 1) * (right - left)
                py = bottom - (y_value - y_low) / (y_high - y_low) * (bottom - top)
                points.append((px, py))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            draw.text((left + 10 + 230 * line_index, top + 8), name, fill=color)
    image.save(path)


def run():
    params = Params()
    rng = np.random.default_rng(params.seed)
    spot_z = feature_normals(params)
    train_paths = qmc_path_count(np.ceil(TRAIN_SCENARIOS_PER_FIT / TRAIN_STATES))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    method_rows = []
    detail_rows = []
    training_rows = []
    plot_payload = {}

    for month in TEST_MONTHS:
        train_states, train_moments = make_states(
            month, params, TRAIN_STATES, spot_z
        )
        validation_states, validation_moments = make_states(
            month, params, VALIDATION_STATES, spot_z, validation=True
        )
        train_values, train_stderr = build_labels(
            train_states,
            train_moments,
            month,
            params,
            rng,
            train_paths,
        )
        benchmark, benchmark_stderr = build_labels(
            validation_states,
            validation_moments,
            month,
            params,
            rng,
            BENCHMARK_PATHS_PER_STATE,
        )
        for variant in VARIANTS:
            for row in range(TRAIN_STATES):
                training_rows.append(
                    {
                        "variant": variant,
                        "month": month,
                        "accrued_return": train_states[0][row],
                        "spot_1": train_states[1][row, 0],
                        "spot_2": train_states[1][row, 1],
                        "spot_3": train_states[1][row, 2],
                        "variance_1": train_states[2][row, 0],
                        "variance_2": train_states[2][row, 1],
                        "variance_3": train_states[2][row, 2],
                        "coupon_mean": train_moments[row][variant][0],
                        "coupon_variance": train_moments[row][variant][1],
                        "coupon_skewness": train_moments[row][variant][2],
                        "coupon_floor_mass": train_moments[row][variant][3],
                        "coupon_cap_mass": train_moments[row][variant][4],
                        "value": train_values[variant][row],
                        "stderr": train_stderr[variant][row],
                    }
                )
            for method in METHODS:
                proxy = fit_proxy(
                    train_states,
                    train_moments,
                    train_values[variant],
                    month,
                    variant,
                    params,
                    method,
                )
                prediction = proxy(validation_states, validation_moments)
                metrics = score(
                    prediction,
                    benchmark[variant],
                    benchmark_stderr[variant],
                )
                method_rows.append(
                    {
                        "variant": variant,
                        "variant_description": VARIANTS[variant],
                        "method": method,
                        "month": month,
                        "remaining_periods": remaining_periods(month, params),
                        "state_dimension": 7,
                        "train_states": TRAIN_STATES,
                        "train_paths_per_state": train_paths,
                        "train_scenarios_used": TRAIN_STATES * train_paths,
                        "validation_states": VALIDATION_STATES,
                        "benchmark_paths_per_state": BENCHMARK_PATHS_PER_STATE,
                        "steps_per_period": STEPS_PER_PERIOD,
                        **metrics,
                        "avg_train_stderr": float(
                            np.mean(train_stderr[variant])
                        ),
                        "avg_benchmark_stderr": float(
                            np.mean(benchmark_stderr[variant])
                        ),
                    }
                )
                for row in range(VALIDATION_STATES):
                    error = prediction[row] - benchmark[variant][row]
                    detail_rows.append(
                        {
                            "variant": variant,
                            "method": method,
                            "month": month,
                            "accrued_return": validation_states[0][row],
                            "spot_1": validation_states[1][row, 0],
                            "spot_2": validation_states[1][row, 1],
                            "spot_3": validation_states[1][row, 2],
                            "variance_1": validation_states[2][row, 0],
                            "variance_2": validation_states[2][row, 1],
                            "variance_3": validation_states[2][row, 2],
                            "benchmark": benchmark[variant][row],
                            "proxy": prediction[row],
                            "signed_error": error,
                            "relative_error": abs(error)
                            / max(
                                abs(benchmark[variant][row]),
                                RELATIVE_ERROR_FLOOR,
                            ),
                            "benchmark_stderr": benchmark_stderr[variant][row],
                        }
                    )
                plot_payload[(variant, month, method)] = (
                    benchmark[variant],
                    prediction,
                )
        print(f"finished reset month {month}")

    write_csv(METHOD_CSV, method_rows)
    write_csv(DETAIL_CSV, detail_rows)
    write_csv(TRAINING_CSV, training_rows)
    aggregate = {}
    best_methods = {}
    for variant in VARIANTS:
        aggregate[variant] = {}
        for method in METHODS:
            rows = [
                row
                for row in method_rows
                if row["variant"] == variant
                and row["method"] == method
                and row["remaining_periods"] > 0
            ]
            aggregate[variant][method] = {
                "worst_max": max(row["max_rel"] for row in rows),
                "avg_p99": float(np.mean([row["p99_rel"] for row in rows])),
                "avg_mae": float(np.mean([row["mae"] for row in rows])),
            }
        # Fixed across variants because development-selected blends did not
        # generalize to untouched state-space designs.
        best_methods[variant] = "sparse_chebyshev"

    lines = [
        "# Three-underlying SLV basket cliquet experiment",
        "",
        "| Variant | Best method | Worst max relative error | Average p99 | Average MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for variant, method in best_methods.items():
        item = aggregate[variant][method]
        lines.append(
            f"| `{variant}` | `{method}` | {100 * item['worst_max']:.3f}% | "
            f"{100 * item['avg_p99']:.3f}% | {item['avg_mae']:.6f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="ascii")

    for variant, method in best_methods.items():
        for month in TEST_MONTHS:
            draw_plot(
                PLOT_DIR / f"{variant}_month_{month:02d}_{method}.png",
                variant,
                month,
                *plot_payload[(variant, month, method)],
            )
    print()
    for variant, method in best_methods.items():
        item = aggregate[variant][method]
        print(
            f"{variant}: {method}, worst max error "
            f"{100 * item['worst_max']:.3f}%"
        )
    print(f"results: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
