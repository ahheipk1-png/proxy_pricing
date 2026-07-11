import csv
from dataclasses import dataclass
from math import exp, log, sqrt
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
    seed: int = 37


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
METHOD_CSV = OUTPUT_DIR / "slv_cliquet_proxy_method_results.csv"
DETAIL_CSV = OUTPUT_DIR / "slv_cliquet_proxy_validation_details.csv"
SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "Markdown"
    / "SLVCliquet"
    / "results"
    / "summary.md"
)

TEST_DAY_INDICES = [0, 3, 6, 9, 12]
TRAIN_STATES = 321
VALIDATION_STATES = 41
TRAIN_SCENARIOS_PER_FIT = 10_000_000
BENCHMARK_PATHS_PER_STATE = 524_288
STEPS_PER_PERIOD = 2
BATCH_PATHS = 131_072
RELATIVE_ERROR_FLOOR = 0.01
RIDGE = 3e-7
LOGIT_EPS = 1e-7
SPOT_RANGE = (65.0, 150.0)
VAR_RANGE = (0.01, 0.12)
METHODS = [
    "logit_accrued_d19",
    "logit_z_d19",
    "logit_slv_anisotropic_d19",
    "logit_slv_local_quadratic",
    "adaptive_hybrid",
]


def radical_inverse(index, base):
    value = 0.0
    fraction = 1.0 / base
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


def remaining_periods(day_index, params):
    return params.n_periods - day_index


def period_dt(params):
    return params.maturity / params.n_periods


def discount(day_index, params):
    return exp(-params.rate * remaining_periods(day_index, params) * period_dt(params))


def payoff(total_return, params):
    return params.notional * np.clip(
        total_return, params.global_floor, params.global_cap
    )


def accrued_range(day_index, params):
    if day_index == 0:
        return 0.0, 0.0
    return day_index * params.local_floor, day_index * params.local_cap


def leverage(spot, params):
    log_moneyness = np.log(np.maximum(spot, 1e-12) / params.s0)
    raw = 1.0 + params.local_skew * np.tanh(log_moneyness / params.local_scale)
    return np.clip(raw, 0.55, 1.45)


def frozen_coupon_moments(spot, variance, params, draws=20001):
    # Deterministic quadrature-like normal grid; used only to construct features.
    u = (np.arange(draws, dtype=float) + 0.5) / draws
    z = np.sqrt(2.0) * inverse_erf(2.0 * u - 1.0)
    sigma = leverage(np.asarray(spot), params) * np.sqrt(np.maximum(variance, 1e-10))
    dt = period_dt(params)
    drift = (params.rate - params.div_yield - 0.5 * sigma**2) * dt
    returns = np.exp(drift[..., None] + sigma[..., None] * sqrt(dt) * z) - 1.0
    clipped = np.clip(returns, params.local_floor, params.local_cap)
    return np.mean(clipped, axis=-1), np.var(clipped, axis=-1)


def inverse_erf(x):
    # Winitzki approximation is ample for feature scaling and avoids scipy.
    x = np.asarray(x, dtype=float)
    a = 0.147
    term = 2.0 / (np.pi * a) + np.log(1.0 - x * x) / 2.0
    return np.sign(x) * np.sqrt(np.sqrt(term * term - np.log(1.0 - x * x) / a) - term)


def exact_tail_value(accrued, day_index, params):
    accrued = np.asarray(accrued, dtype=float)
    m = remaining_periods(day_index, params)
    if m == 0:
        return payoff(accrued, params)
    min_total = accrued + m * params.local_floor
    max_total = accrued + m * params.local_cap
    values = np.full_like(accrued, np.nan)
    values[min_total >= params.global_cap] = (
        discount(day_index, params) * params.notional * params.global_cap
    )
    values[max_total <= params.global_floor] = (
        discount(day_index, params) * params.notional * params.global_floor
    )
    return values


def structured_features(accrued, spot, variance, day_index, params):
    accrued = np.asarray(accrued, dtype=float)
    spot = np.asarray(spot, dtype=float)
    variance = np.asarray(variance, dtype=float)
    m = remaining_periods(day_index, params)
    if m == 0:
        zeros = np.zeros_like(accrued)
        return zeros, zeros, zeros
    coupon_mean, coupon_var = frozen_coupon_moments(spot, variance, params, draws=1001)
    expected_total = accrued + m * coupon_mean
    midpoint = 0.5 * (params.global_floor + params.global_cap)
    z = (expected_total - midpoint) / np.sqrt(np.maximum(m * coupon_var, 1e-8))
    log_spot = np.log(np.maximum(spot, 1e-12) / params.s0)
    log_var = np.log(np.maximum(variance, 1e-8) / params.theta)
    return z, log_spot, log_var


def make_states(
    day_index, params, n_states, validation=False, validation_offset=None
):
    low, high = accrued_range(day_index, params)
    accrued = np.empty(n_states)
    spot = np.empty(n_states)
    variance = np.empty(n_states)
    offset = (
        (5003 if validation_offset is None else validation_offset)
        if validation
        else 0
    )
    for idx in range(n_states):
        i = idx + 1 + offset
        u_a = radical_inverse(i, 2)
        u_s = radical_inverse(i, 3)
        u_v = radical_inverse(i, 5)
        spot[idx] = SPOT_RANGE[0] * (SPOT_RANGE[1] / SPOT_RANGE[0]) ** u_s
        variance[idx] = VAR_RANGE[0] * (VAR_RANGE[1] / VAR_RANGE[0]) ** u_v
        if high <= low:
            accrued[idx] = low
            continue
        if idx % 4 == 0:
            accrued[idx] = low + u_a * (high - low)
        else:
            m = remaining_periods(day_index, params)
            mean, var = frozen_coupon_moments(
                np.array([spot[idx]]), np.array([variance[idx]]), params, draws=1001
            )
            boundary = (
                params.global_floor
                if idx % 4 in {1, 2}
                else params.global_cap
            )
            center = boundary - m * float(mean[0])
            width = max(sqrt(max(m * float(var[0]), 1e-8)), (high - low) / 40.0)
            accrued[idx] = np.clip(
                center + (2.0 * u_a - 1.0) * 5.0 * width, low, high
            )
    if validation and n_states >= 9:
        corner_spots = [SPOT_RANGE[0], params.s0, SPOT_RANGE[1]]
        corner_vars = [VAR_RANGE[0], params.theta, VAR_RANGE[1]]
        for idx, (s, v) in enumerate(
            (pair for s in corner_spots for pair in [(s, x) for x in corner_vars])
        ):
            spot[idx], variance[idx] = s, v
            accrued[idx] = low + 0.5 * (high - low)
    return accrued, spot, variance


def simulate_value(accrued, spot0, variance0, day_index, params, rng, n_paths):
    exact = exact_tail_value(np.array([accrued]), day_index, params)[0]
    if np.isfinite(exact):
        return float(exact), 0.0
    m = remaining_periods(day_index, params)
    n_steps = m * STEPS_PER_PERIOD
    dt = period_dt(params) / STEPS_PER_PERIOD
    df = discount(day_index, params)
    count = 0
    total = 0.0
    total_sq = 0.0
    rho_scale = sqrt(1.0 - params.rho**2)
    coupon_mean, _ = frozen_coupon_moments(
        np.array([spot0]), np.array([variance0]), params, draws=1001
    )
    expected_total = accrued + m * float(coupon_mean[0])
    required_coupon = max(
        (params.global_floor - expected_total) / max(m, 1), 0.0
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

    seed = rng.integers(0, np.iinfo(np.int64).max)
    for base_normals in sobol_antithetic_batches(
        n_paths, 2 * n_steps, seed, BATCH_PATHS
    ):
        batch = len(base_normals)
        z1 = base_normals[:, :n_steps]
        z2 = importance_shift + base_normals[:, n_steps:]
        likelihood = np.exp(
            -importance_shift * np.sum(z2, axis=1)
            + 0.5 * n_steps * importance_shift**2
        )
        spot = np.full(batch, spot0)
        variance = np.full(batch, variance0)
        period_start = spot.copy()
        future_sum = np.zeros(batch)

        for step in range(n_steps):
            variance_pos = np.maximum(variance, 0.0)
            root_v = np.sqrt(variance_pos)
            z_var = z1[:, step]
            z_spot = params.rho * z_var + rho_scale * z2[:, step]
            lev = leverage(spot, params)
            inst_var = lev * lev * variance_pos
            spot *= np.exp(
                (params.rate - params.div_yield - 0.5 * inst_var) * dt
                + lev * root_v * sqrt(dt) * z_spot
            )
            variance += (
                params.kappa * (params.theta - variance_pos) * dt
                + params.vol_of_var * root_v * sqrt(dt) * z_var
            )
            variance = np.maximum(variance, 0.0)
            if (step + 1) % STEPS_PER_PERIOD == 0:
                period_return = spot / period_start - 1.0
                future_sum += np.clip(
                    period_return, params.local_floor, params.local_cap
                )
                period_start = spot.copy()

        values = df * payoff(accrued + future_sum, params) * likelihood
        count += batch
        total += float(np.sum(values))
        total_sq += float(np.sum(values * values))

    mean = total / count
    variance_of_values = max(
        (total_sq - count * mean * mean) / max(count - 1, 1), 0.0
    )
    return mean, sqrt(variance_of_values / count)


def build_labels(states, day_index, params, rng, paths_per_state):
    accrued, spot, variance = states
    values = np.empty(len(accrued))
    stderr = np.empty(len(accrued))
    common_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    for idx in range(len(accrued)):
        values[idx], stderr[idx] = simulate_value(
            accrued[idx],
            spot[idx],
            variance[idx],
            day_index,
            params,
            np.random.default_rng(common_seed),
            paths_per_state,
        )
    return values, stderr


def scale(values):
    values = np.asarray(values)
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low + 1e-12:
        return np.zeros_like(values), low, high
    return 2.0 * (values - low) / (high - low) - 1.0, low, high


def apply_scale(values, low, high):
    if high <= low + 1e-12:
        return np.zeros_like(values)
    return np.clip(2.0 * (values - low) / (high - low) - 1.0, -1.0, 1.0)


def anisotropic_design(z, log_spot, log_var, degree=19):
    tz = chebvander(z, degree)
    ts = chebvander(log_spot, 3)
    tv = chebvander(log_var, 3)
    columns = []
    for i in range(degree + 1):
        for j in range(4):
            for k in range(4):
                if i + 5 * j + 5 * k <= degree:
                    columns.append(tz[:, i] * ts[:, j] * tv[:, k])
    return np.column_stack(columns)


def ridge_solve(design, target):
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def fit_proxy(states, values, day_index, params, method):
    if method == "adaptive_hybrid":
        component = (
            "logit_slv_local_quadratic"
            if remaining_periods(day_index, params) >= 9
            else "logit_slv_anisotropic_d19"
        )
        return fit_proxy(states, values, day_index, params, component)
    accrued, spot, variance = states
    exact = exact_tail_value(accrued, day_index, params)
    mask = ~np.isfinite(exact)
    if not np.any(mask):
        return lambda new_states: exact_tail_value(new_states[0], day_index, params)
    lower = discount(day_index, params) * params.notional * params.global_floor
    upper = discount(day_index, params) * params.notional * params.global_cap
    normalized = np.clip(
        (values[mask] - lower) / max(upper - lower, 1e-12),
        LOGIT_EPS,
        1.0 - LOGIT_EPS,
    )
    target = np.log(normalized / (1.0 - normalized))

    if method == "logit_accrued_d19":
        x, x_low, x_high = scale(accrued[mask])
        coeffs = ridge_solve(chebvander(x, 19), target)

        def design(new_states):
            return chebvander(apply_scale(new_states[0], x_low, x_high), 19)

    elif method == "logit_z_d19":
        z, _, _ = structured_features(
            accrued[mask], spot[mask], variance[mask], day_index, params
        )
        x, x_low, x_high = scale(z)
        coeffs = ridge_solve(chebvander(x, 19), target)

        def design(new_states):
            new_z, _, _ = structured_features(*new_states, day_index, params)
            return chebvander(apply_scale(new_z, x_low, x_high), 19)

    elif method == "logit_slv_anisotropic_d19":
        z, s, v = structured_features(
            accrued[mask], spot[mask], variance[mask], day_index, params
        )
        z_scaled, z_low, z_high = scale(z)
        s_scaled, s_low, s_high = scale(s)
        v_scaled, v_low, v_high = scale(v)
        coeffs = ridge_solve(
            anisotropic_design(z_scaled, s_scaled, v_scaled), target
        )

        def design(new_states):
            new_z, new_s, new_v = structured_features(
                *new_states, day_index, params
            )
            return anisotropic_design(
                apply_scale(new_z, z_low, z_high),
                apply_scale(new_s, s_low, s_high),
                apply_scale(new_v, v_low, v_high),
            )

    elif method == "logit_slv_local_quadratic":
        z, s, v = structured_features(
            accrued[mask], spot[mask], variance[mask], day_index, params
        )
        z_scaled, z_low, z_high = scale(z)
        s_scaled, s_low, s_high = scale(s)
        v_scaled, v_low, v_high = scale(v)
        train_feature = np.column_stack((z_scaled, s_scaled, v_scaled))
        train_target = target.copy()
        coeffs = None

        def design(new_states):
            new_z, new_s, new_v = structured_features(
                *new_states, day_index, params
            )
            query = np.column_stack(
                (
                    apply_scale(new_z, z_low, z_high),
                    apply_scale(new_s, s_low, s_high),
                    apply_scale(new_v, v_low, v_high),
                )
            )
            fitted = np.empty(len(query))
            neighbor_count = min(64, len(train_feature))
            for row_idx, point in enumerate(query):
                delta = train_feature - point
                distance_sq = np.sum(delta * delta, axis=1)
                neighbors = np.argpartition(
                    distance_sq, neighbor_count - 1
                )[:neighbor_count]
                local_delta = delta[neighbors]
                bandwidth = max(float(np.max(distance_sq[neighbors])), 1e-8)
                weights = np.exp(-3.0 * distance_sq[neighbors] / bandwidth)
                dz, ds, dv = local_delta.T
                local_design = np.column_stack(
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
                weighted_design = local_design * root_weight[:, None]
                weighted_target = train_target[neighbors] * root_weight
                local_penalty = np.eye(local_design.shape[1]) * 1e-5
                local_penalty[0, 0] = 0.0
                local_coeffs = np.linalg.solve(
                    weighted_design.T @ weighted_design + local_penalty,
                    weighted_design.T @ weighted_target,
                )
                fitted[row_idx] = local_coeffs[0]
            # Prediction machinery expects a design matrix times coeffs.
            return fitted[:, None]

    else:
        raise ValueError(method)

    def predict(new_states):
        new_accrued = new_states[0]
        exact_new = exact_tail_value(new_accrued, day_index, params)
        output = np.empty_like(new_accrued)
        exact_mask = np.isfinite(exact_new)
        output[exact_mask] = exact_new[exact_mask]
        if np.any(~exact_mask):
            fitted_design = design(tuple(x[~exact_mask] for x in new_states))
            raw = (
                fitted_design[:, 0]
                if method == "logit_slv_local_quadratic"
                else fitted_design @ coeffs
            )
            bounded = 1.0 / (1.0 + np.exp(-np.clip(raw, -35.0, 35.0)))
            output[~exact_mask] = lower + (upper - lower) * bounded
        return np.clip(output, lower, upper)

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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_plot(path, day_index, accrued, benchmark, prediction):
    if Image is None:
        return
    width, height = 1200, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panels = [(55, 70, 1140, 350), (55, 390, 1140, 650), (55, 690, 1140, 940)]
    order = np.argsort(benchmark)
    x = np.arange(len(benchmark), dtype=float)
    signed_error = prediction - benchmark
    signed_relative = (
        100.0 * signed_error / np.maximum(np.abs(benchmark), RELATIVE_ERROR_FLOOR)
    )
    series = [
        [("benchmark", benchmark[order], (35, 90, 180)), ("proxy", prediction[order], (210, 55, 55))],
        [("signed error", signed_error[order], (25, 145, 85))],
        [("signed relative error (%)", signed_relative[order], (125, 70, 170))],
    ]
    draw.text(
        (55, 20),
        f"SLV cliquet, reset month {day_index}; validation states ranked by benchmark",
        fill="black",
    )
    for panel, lines in zip(panels, series):
        left, top, right, bottom = panel
        draw.rectangle(panel, outline=(180, 180, 180))
        all_y = np.concatenate([line[1] for line in lines])
        y_low, y_high = float(np.min(all_y)), float(np.max(all_y))
        if y_high <= y_low:
            y_high = y_low + 1.0
        x_low, x_high = float(np.min(x)), float(np.max(x))
        if x_high <= x_low:
            x_high = x_low + 1.0
        if y_low < 0.0 < y_high:
            y0 = bottom - (0.0 - y_low) / (y_high - y_low) * (bottom - top)
            draw.line((left, y0, right, y0), fill=(210, 210, 210))
        for line_idx, (name, y, color) in enumerate(lines):
            points = []
            for xv, yv in zip(x, y):
                px = left + (xv - x_low) / (x_high - x_low) * (right - left)
                py = bottom - (yv - y_low) / (y_high - y_low) * (bottom - top)
                points.append((px, py))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            draw.text((left + 10 + 170 * line_idx, top + 8), name, fill=color)
    image.save(path)


def run():
    params = Params()
    rng = np.random.default_rng(params.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    method_rows = []
    detail_rows = []
    plot_data = {}

    for day_index in TEST_DAY_INDICES:
        train_states = make_states(day_index, params, TRAIN_STATES)
        validation_states = make_states(
            day_index, params, VALIDATION_STATES, validation=True
        )
        train_paths = qmc_path_count(np.ceil(TRAIN_SCENARIOS_PER_FIT / TRAIN_STATES))
        train_values, train_stderr = build_labels(
            train_states, day_index, params, rng, train_paths
        )
        benchmark, benchmark_stderr = build_labels(
            validation_states,
            day_index,
            params,
            rng,
            BENCHMARK_PATHS_PER_STATE,
        )

        for method in METHODS:
            proxy = fit_proxy(train_states, train_values, day_index, params, method)
            prediction = proxy(validation_states)
            metrics = score(prediction, benchmark, benchmark_stderr)
            method_rows.append(
                {
                    "method": method,
                    "day_index": day_index,
                    "remaining_periods": remaining_periods(day_index, params),
                    "train_states": TRAIN_STATES,
                    "train_paths_per_state": train_paths,
                    "train_scenarios_used": train_paths * TRAIN_STATES,
                    "benchmark_paths_per_state": BENCHMARK_PATHS_PER_STATE,
                    "validation_states": VALIDATION_STATES,
                    "steps_per_period": STEPS_PER_PERIOD,
                    **metrics,
                    "avg_train_stderr": float(np.mean(train_stderr)),
                    "avg_benchmark_stderr": float(np.mean(benchmark_stderr)),
                }
            )
            z, _, _ = structured_features(
                *validation_states, day_index, params
            )
            for idx in range(VALIDATION_STATES):
                error = prediction[idx] - benchmark[idx]
                detail_rows.append(
                    {
                        "method": method,
                        "day_index": day_index,
                        "accrued_return": validation_states[0][idx],
                        "spot": validation_states[1][idx],
                        "variance": validation_states[2][idx],
                        "expected_total_z": z[idx],
                        "benchmark": benchmark[idx],
                        "proxy": prediction[idx],
                        "signed_error": error,
                        "relative_error": abs(error)
                        / max(abs(benchmark[idx]), RELATIVE_ERROR_FLOOR),
                        "benchmark_stderr": benchmark_stderr[idx],
                    }
                )
            plot_data[(day_index, method)] = (
                validation_states[0],
                benchmark,
                prediction,
            )
        print(f"finished reset month {day_index}")

    write_csv(METHOD_CSV, method_rows)
    write_csv(DETAIL_CSV, detail_rows)
    nonterminal = [row for row in method_rows if row["remaining_periods"] > 0]
    aggregate = {}
    for method in METHODS:
        rows = [row for row in nonterminal if row["method"] == method]
        aggregate[method] = {
            "worst_max": max(row["max_rel"] for row in rows),
            "avg_p99": float(np.mean([row["p99_rel"] for row in rows])),
            "avg_mae": float(np.mean([row["mae"] for row in rows])),
        }
    best = min(METHODS, key=lambda name: aggregate[name]["worst_max"])
    summary = [
        "# SLV cliquet proxy experiment",
        "",
        f"Best method: `{best}`.",
        "",
        "| Method | Worst max relative error | Average p99 | Average MAE |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        item = aggregate[method]
        summary.append(
            f"| `{method}` | {100 * item['worst_max']:.3f}% | "
            f"{100 * item['avg_p99']:.3f}% | {item['avg_mae']:.6f} |"
        )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="ascii")
    for day_index in TEST_DAY_INDICES:
        draw_plot(
            PLOT_DIR / f"slv_cliquet_day_{day_index:02d}_{best}.png",
            day_index,
            *plot_data[(day_index, best)],
        )
    print()
    print(f"best method: {best}")
    print(f"worst max relative error: {100 * aggregate[best]['worst_max']:.3f}%")
    print(f"results: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
