import csv
from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm, qmc

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


@dataclass(frozen=True)
class Params:
    n_assets: int = 10
    strike: float = 100.0
    rate: float = 0.05
    maturity: float = 1.0
    n_fixings: int = 12
    seed: int = 137


TEST_DAY_INDICES = [0, 3, 6, 9, 10, 11]
TRAIN_STATES = 513
VALIDATION_STATES = 121
TRAIN_SCENARIOS_PER_FIT = 20_000_000
BENCHMARK_PATHS_PER_STATE = 500_000
SIMULATION_BATCH_PATHS = 8_192
RELATIVE_ERROR_FLOOR = 0.05
LOG_EPS = 1e-10
RIDGE = 3e-6
OUTPUT_DIR = Path(__file__).resolve().parent / "BasketAsianOptExperiment" / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
METHOD_CSV = OUTPUT_DIR / "basket_asian_proxy_method_results.csv"
DETAIL_CSV = OUTPUT_DIR / "basket_asian_proxy_validation_details.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"

METHODS = [
    "moment_lognormal",
    "relative_residual_sparse_chebyshev_pca",
    "log_factor_sparse_chebyshev_pca",
    "pchip_calibrated_log_factor_pca",
    "residual_sparse_chebyshev_pca",
    "blend_correction_sparse_chebyshev_pca",
]


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def power_of_two_at_least(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 1))))


def qmc_path_count(target_paths):
    return power_of_two_at_least(target_paths)


def sobol_batches(target_paths, dimension, seed, max_batch):
    total = qmc_path_count(target_paths)
    batch_size = min(power_of_two_at_least(max_batch), total)
    engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    remaining = total
    while remaining:
        batch = min(batch_size, remaining)
        uniforms = engine.random(batch)
        uniforms = np.clip(uniforms, 1e-12, 1.0 - 1e-12)
        yield norm.ppf(uniforms)
        remaining -= batch


def asset_setup(params):
    div_yields = np.linspace(0.010, 0.035, params.n_assets)
    vols = np.array([0.18, 0.21, 0.24, 0.19, 0.27, 0.31, 0.23, 0.29, 0.34, 0.26])
    s0 = np.full(params.n_assets, 100.0)
    return s0, div_yields, vols


def correlation_matrix(params):
    loadings = np.array(
        [
            [0.72, 0.18, 0.05],
            [0.68, 0.25, -0.10],
            [0.63, -0.20, 0.18],
            [0.55, -0.30, -0.15],
            [-0.50, 0.45, 0.10],
            [-0.58, 0.35, -0.20],
            [-0.42, -0.50, 0.25],
            [0.25, 0.62, -0.28],
            [0.10, -0.68, -0.35],
            [-0.18, 0.20, 0.72],
        ],
        dtype=float,
    )
    raw = loadings @ loadings.T
    diag_extra = np.linspace(0.28, 0.46, params.n_assets)
    raw += np.diag(diag_extra)
    scale = np.sqrt(np.diag(raw))
    corr = raw / scale[:, None] / scale[None, :]
    return np.clip(corr, -0.85, 0.95)


def pca_loadings(params):
    _, _, vols = asset_setup(params)
    cov = np.outer(vols, vols) * correlation_matrix(params)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    return eigval[order], eigvec[:, order]


def daily_dt(params):
    return params.maturity / (params.n_fixings - 1)


def future_count(day_index, params):
    return params.n_fixings - day_index - 1


def discount(day_index, params):
    return exp(-params.rate * future_count(day_index, params) * daily_dt(params))


def basket_level(spots):
    return np.mean(np.asarray(spots, dtype=float), axis=-1)


def payoff_from_average(avg, params):
    return np.maximum(np.asarray(avg, dtype=float) - params.strike, 0.0)


def precompute_future_moments(day_index, params):
    _, divs, vols = asset_setup(params)
    corr = correlation_matrix(params)
    m = future_count(day_index, params)
    weights = np.full(params.n_assets, 1.0 / params.n_assets)
    if m == 0:
        return np.zeros(params.n_assets), np.zeros((params.n_assets, params.n_assets))
    dt = daily_dt(params)
    times = np.arange(1, m + 1, dtype=float) * dt
    mu = params.rate - divs
    mean_coeff = weights * np.array([np.sum(np.exp(mu[i] * times)) for i in range(params.n_assets)])
    cov_coeff = np.zeros((params.n_assets, params.n_assets))
    for i in range(params.n_assets):
        for k in range(params.n_assets):
            total = 0.0
            for tj in times:
                for tl in times:
                    cross = exp(
                        mu[i] * tj
                        + mu[k] * tl
                        + corr[i, k] * vols[i] * vols[k] * min(tj, tl)
                    )
                    product = exp(mu[i] * tj) * exp(mu[k] * tl)
                    total += cross - product
            cov_coeff[i, k] = weights[i] * weights[k] * total
    return mean_coeff, cov_coeff


def final_average_moments(spots, running_sum_before, day_index, params):
    spots = np.asarray(spots, dtype=float)
    running_sum_before = np.asarray(running_sum_before, dtype=float)
    mean_coeff, cov_coeff = precompute_future_moments(day_index, params)
    current_sum = running_sum_before + basket_level(spots)
    future_mean = spots @ mean_coeff
    future_var = np.einsum("ni,ij,nj->n", spots, cov_coeff, spots)
    mean_avg = (current_sum + future_mean) / params.n_fixings
    var_avg = np.maximum(future_var, 0.0) / (params.n_fixings * params.n_fixings)
    return mean_avg, var_avg


def exact_linear_value(spots, running_sum_before, day_index, params):
    mean_avg, _ = final_average_moments(spots, running_sum_before, day_index, params)
    return discount(day_index, params) * (mean_avg - params.strike)


def linear_tail_mask(spots, running_sum_before, params):
    current_sum = np.asarray(running_sum_before, dtype=float) + basket_level(spots)
    return current_sum >= params.n_fixings * params.strike


def moment_lognormal_value(spots, running_sum_before, day_index, params):
    spots = np.asarray(spots, dtype=float)
    running_sum_before = np.asarray(running_sum_before, dtype=float)
    m = future_count(day_index, params)
    current_sum = running_sum_before + basket_level(spots)
    if m == 0:
        return payoff_from_average(current_sum / params.n_fixings, params)
    linear = linear_tail_mask(spots, running_sum_before, params)
    mean_avg, var_avg = final_average_moments(spots, running_sum_before, day_index, params)
    output = np.empty_like(mean_avg)
    output[linear] = exact_linear_value(spots[linear], running_sum_before[linear], day_index, params)
    active = ~linear
    if np.any(active):
        mean = np.maximum(mean_avg[active], 1e-12)
        variance = np.maximum(var_avg[active], 0.0)
        small = variance <= 1e-14
        active_value = np.empty_like(mean)
        active_value[small] = discount(day_index, params) * np.maximum(mean[small] - params.strike, 0.0)
        if np.any(~small):
            vol2 = np.log1p(variance[~small] / np.maximum(mean[~small] * mean[~small], 1e-24))
            vol = np.sqrt(np.maximum(vol2, 1e-16))
            d1 = (np.log(mean[~small] / params.strike) + 0.5 * vol2) / vol
            d2 = d1 - vol
            active_value[~small] = discount(day_index, params) * (
                mean[~small] * normal_cdf(d1) - params.strike * normal_cdf(d2)
            )
        output[active] = active_value
    return np.maximum(output, 0.0)


def make_states(day_index, params, count, validation=False):
    s0, _, _ = asset_setup(params)
    _, eigvec = pca_loadings(params)
    dimension = 6 if day_index > 0 else 5
    engine = qmc.Sobol(d=dimension, scramble=True, seed=params.seed + 1000 * day_index + (17 if validation else 0))
    n = power_of_two_at_least(count)
    u = engine.random_base2(int(np.log2(n)))[:count]
    basket_low, basket_high = log(0.58), log(1.58)
    if validation:
        basket_low, basket_high = log(0.62), log(1.52)
    basket_target = params.strike * np.exp(basket_low + u[:, 0] * (basket_high - basket_low))
    pc_scale = np.array([0.42, 0.34, 0.27, 0.22])
    scores = (2.0 * u[:, 1:5] - 1.0) * pc_scale
    log_relative = scores @ eigvec[:, :4].T
    raw = s0 * np.exp(log_relative)
    spots = raw * (basket_target / basket_level(raw))[:, None]
    if day_index == 0:
        running_sum = np.zeros(count)
    else:
        run_low, run_high = log(0.62), log(1.42)
        if validation:
            run_low, run_high = log(0.66), log(1.36)
        running_avg = params.strike * np.exp(run_low + u[:, 5] * (run_high - run_low))
        running_sum = day_index * running_avg
    if count >= 9:
        base_levels = np.linspace(65.0, 150.0, min(9, count))
        for idx, level in enumerate(base_levels):
            spots[idx] = level
            running_sum[idx] = 0.0 if day_index == 0 else day_index * params.strike
    return spots.astype(float), running_sum.astype(float)


def simulate_labels(spots, running_sum_before, day_index, params, paths, seed):
    spots = np.asarray(spots, dtype=float)
    running_sum_before = np.asarray(running_sum_before, dtype=float)
    count_states = len(spots)
    m = future_count(day_index, params)
    current_sum = running_sum_before + basket_level(spots)
    if m == 0:
        values = payoff_from_average(current_sum / params.n_fixings, params)
        return values, np.zeros_like(values)
    values = np.zeros(count_states)
    values_sq = np.zeros(count_states)
    counts = 0
    linear = linear_tail_mask(spots, running_sum_before, params)
    values[linear] = exact_linear_value(spots[linear], running_sum_before[linear], day_index, params)
    active = ~linear
    if not np.any(active):
        return values, np.zeros_like(values)
    active_spots = spots[active]
    active_current = current_sum[active]
    _, divs, vols = asset_setup(params)
    corr = correlation_matrix(params)
    chol = np.linalg.cholesky(corr)
    dt = daily_dt(params)
    drift = (params.rate - divs - 0.5 * vols * vols) * dt
    diffusion = vols * sqrt(dt)
    dimension = m * params.n_assets
    df = discount(day_index, params)
    for normals in sobol_batches(paths, dimension, seed, SIMULATION_BATCH_PATHS):
        batch = len(normals)
        z = normals.reshape(batch, m, params.n_assets) @ chol.T
        increments = drift[None, None, :] + diffusion[None, None, :] * z
        growth = np.exp(np.cumsum(increments, axis=1))
        growth_sum = np.sum(growth, axis=1)
        future_sum = growth_sum @ active_spots.T / params.n_assets
        average = (active_current[None, :] + future_sum) / params.n_fixings
        samples = df * payoff_from_average(average, params)
        values[active] += np.sum(samples, axis=0)
        values_sq[active] += np.sum(samples * samples, axis=0)
        counts += batch
    mean = values.copy()
    mean[active] = values[active] / counts
    variance = np.zeros(count_states)
    variance[active] = np.maximum(
        (values_sq[active] - counts * mean[active] * mean[active]) / max(counts - 1, 1),
        0.0,
    )
    stderr = np.zeros(count_states)
    stderr[active] = np.sqrt(variance[active] / counts)
    return mean, stderr


def feature_matrix(spots, running_sum_before, day_index, params):
    s0, _, _ = asset_setup(params)
    eigval, eigvec = pca_loadings(params)
    log_spot = np.log(np.maximum(spots, 1e-12) / s0)
    pc = log_spot @ eigvec[:, :4]
    basket = basket_level(spots)
    running_avg = (
        np.full_like(basket, params.strike)
        if day_index == 0
        else np.asarray(running_sum_before, dtype=float) / day_index
    )
    mean_avg, var_avg = final_average_moments(spots, running_sum_before, day_index, params)
    effective_vol = np.sqrt(np.log1p(var_avg / np.maximum(mean_avg * mean_avg, 1e-24)))
    base = moment_lognormal_value(spots, running_sum_before, day_index, params)
    dispersion = np.sqrt(np.mean((log_spot - np.mean(log_spot, axis=1, keepdims=True)) ** 2, axis=1))
    moneyness = np.log(np.maximum(mean_avg, 1e-12) / params.strike)
    features = np.column_stack(
        [
            moneyness,
            effective_vol,
            np.log(np.maximum(basket, 1e-12) / params.strike),
            np.log(np.maximum(running_avg, 1e-12) / params.strike),
            dispersion,
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            pc[:, 3],
            np.log1p(np.maximum(base, 0.0)),
        ]
    )
    return features


def scale_features(features):
    low = np.quantile(features, 0.01, axis=0)
    high = np.quantile(features, 0.99, axis=0)
    width = np.maximum(high - low, 1e-10)
    return np.clip(2.0 * (features - low) / width - 1.0, -1.5, 1.5), low, width


def apply_feature_scale(features, low, width):
    return np.clip(2.0 * (features - low) / width - 1.0, -1.5, 1.5)


def sparse_chebyshev_design(features):
    columns = [np.ones(len(features))]
    vanders = [chebvander(features[:, i], 5) for i in range(features.shape[1])]
    for i, vander in enumerate(vanders):
        max_degree = 5 if i <= 1 else 3
        for degree in range(1, max_degree + 1):
            columns.append(vander[:, degree])
    for i in range(features.shape[1]):
        for j in range(i + 1, features.shape[1]):
            if i <= 1 or j <= 4:
                columns.append(features[:, i] * features[:, j])
    for i in range(2, features.shape[1]):
        columns.append(features[:, 0] * features[:, i] * features[:, i])
    return np.column_stack(columns)


def ridge_solve(design, target, ridge):
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def binned_pchip(x, y, bins=45):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    chunks = np.array_split(np.arange(len(x)), min(bins, len(x)))
    centers = []
    levels = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        centers.append(float(np.mean(x[chunk])))
        levels.append(float(np.mean(y[chunk])))
    centers = np.asarray(centers)
    levels = np.asarray(levels)
    unique_centers, unique_indices = np.unique(centers, return_index=True)
    unique_levels = levels[unique_indices]
    if len(unique_centers) < 2:
        const = float(unique_levels[0]) if len(unique_levels) else 0.0
        return lambda new_x: np.full_like(np.asarray(new_x, dtype=float), const)
    interpolator = PchipInterpolator(unique_centers, unique_levels, extrapolate=False)

    def predict(new_x):
        new_x = np.asarray(new_x, dtype=float)
        clipped = np.clip(new_x, unique_centers[0], unique_centers[-1])
        return interpolator(clipped)

    return predict


def fit_proxy(spots, running_sum, values, day_index, params, method):
    base = moment_lognormal_value(spots, running_sum, day_index, params)
    if method == "moment_lognormal":
        return lambda new_spots, new_running: moment_lognormal_value(new_spots, new_running, day_index, params)
    features = feature_matrix(spots, running_sum, day_index, params)
    scaled, low, width = scale_features(features)
    design = sparse_chebyshev_design(scaled)
    if method == "residual_sparse_chebyshev_pca":
        coeffs = ridge_solve(design, values - base, RIDGE)

        def predict(new_spots, new_running):
            new_base = moment_lognormal_value(new_spots, new_running, day_index, params)
            new_features = apply_feature_scale(feature_matrix(new_spots, new_running, day_index, params), low, width)
            return np.maximum(new_base + sparse_chebyshev_design(new_features) @ coeffs, 0.0)

        return predict
    if method == "relative_residual_sparse_chebyshev_pca":
        denom = np.maximum(np.maximum(base, values), RELATIVE_ERROR_FLOOR)
        target = (values - base) / denom
        coeffs = ridge_solve(design, target, 20.0 * RIDGE)

        def predict(new_spots, new_running):
            new_base = moment_lognormal_value(new_spots, new_running, day_index, params)
            new_denom = np.maximum(new_base, RELATIVE_ERROR_FLOOR)
            new_features = apply_feature_scale(feature_matrix(new_spots, new_running, day_index, params), low, width)
            correction = np.clip(sparse_chebyshev_design(new_features) @ coeffs, -0.40, 0.40)
            return np.maximum(new_base + new_denom * correction, 0.0)

        return predict
    if method == "log_factor_sparse_chebyshev_pca":
        eps = RELATIVE_ERROR_FLOOR
        target = np.log((np.maximum(values, 0.0) + eps) / (np.maximum(base, 0.0) + eps))
        coeffs = ridge_solve(design, target, 20.0 * RIDGE)

        def predict(new_spots, new_running):
            new_base = moment_lognormal_value(new_spots, new_running, day_index, params)
            new_features = apply_feature_scale(feature_matrix(new_spots, new_running, day_index, params), low, width)
            factor = np.exp(np.clip(sparse_chebyshev_design(new_features) @ coeffs, -0.35, 0.35))
            return np.maximum((new_base + eps) * factor - eps, 0.0)

        return predict
    if method == "pchip_calibrated_log_factor_pca":
        factor_proxy = fit_proxy(spots, running_sum, values, day_index, params, "log_factor_sparse_chebyshev_pca")
        fitted = factor_proxy(spots, running_sum)
        denom = np.maximum(np.maximum(values, fitted), RELATIVE_ERROR_FLOOR)
        residual_ratio = np.clip((values - fitted) / denom, -0.25, 0.25)
        calibration = binned_pchip(features[:, 0], residual_ratio)

        def predict(new_spots, new_running):
            raw = factor_proxy(new_spots, new_running)
            new_features = feature_matrix(new_spots, new_running, day_index, params)
            new_denom = np.maximum(raw, RELATIVE_ERROR_FLOOR)
            correction = np.clip(calibration(new_features[:, 0]), -0.18, 0.18)
            return np.maximum(raw + new_denom * correction, 0.0)

        return predict
    if method == "blend_correction_sparse_chebyshev_pca":
        factor_proxy = fit_proxy(spots, running_sum, values, day_index, params, "log_factor_sparse_chebyshev_pca")
        residual_proxy = fit_proxy(spots, running_sum, values, day_index, params, "residual_sparse_chebyshev_pca")

        def predict(new_spots, new_running):
            return 0.55 * residual_proxy(new_spots, new_running) + 0.45 * factor_proxy(new_spots, new_running)

        return predict
    raise ValueError(method)


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
        "median_error_over_benchmark_se": float(np.median(absolute / np.maximum(stderr, 1e-12))),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_plot(path, day_index, benchmark, prediction):
    if Image is None:
        return
    order = np.argsort(benchmark)
    benchmark = benchmark[order]
    prediction = prediction[order]
    error = prediction - benchmark
    rel = 100.0 * error / np.maximum(np.abs(benchmark), RELATIVE_ERROR_FLOOR)
    image = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 24), f"10-asset basket Asian validation, day {day_index}; states ranked by benchmark", fill=(0, 0, 0))
    panels = [(55, 70, 1150, 325), (55, 365, 1150, 600), (55, 640, 1150, 860)]
    series = [
        [("benchmark", benchmark, (0, 0, 0)), ("proxy", prediction, (210, 40, 40))],
        [("signed error", error, (20, 130, 75))],
        [("signed relative error (%)", rel, (35, 80, 190))],
    ]
    x = np.arange(len(benchmark))
    for panel, lines in zip(panels, series):
        left, top, right, bottom = panel
        draw.rectangle(panel, outline=(185, 185, 185))
        all_y = np.concatenate([line[1] for line in lines])
        y_low, y_high = float(np.min(all_y)), float(np.max(all_y))
        if abs(y_high - y_low) < 1e-12:
            y_high = y_low + 1.0
        margin = max(0.06 * (y_high - y_low), 1e-8)
        y_low, y_high = y_low - margin, y_high + margin
        if y_low < 0.0 < y_high:
            y0 = bottom - (0.0 - y_low) / (y_high - y_low) * (bottom - top)
            draw.line((left, y0, right, y0), fill=(220, 220, 220))
        for idx, (name, values, color) in enumerate(lines):
            points = []
            for x_value, y_value in zip(x, values):
                px = left + x_value / max(len(x) - 1, 1) * (right - left)
                py = bottom - (y_value - y_low) / (y_high - y_low) * (bottom - top)
                points.append((px, py))
            if len(points) > 1:
                draw.line(points, fill=color, width=3 if idx == 0 else 2)
            draw.text((left + 10 + 190 * idx, top + 8), name, fill=color)
    image.save(path)


def run():
    params = Params()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    method_rows = []
    detail_rows = []
    train_paths = qmc_path_count(np.ceil(TRAIN_SCENARIOS_PER_FIT / TRAIN_STATES))
    benchmark_paths = qmc_path_count(BENCHMARK_PATHS_PER_STATE)
    best_predictions = {}
    for day_index in TEST_DAY_INDICES:
        train_spots, train_running = make_states(day_index, params, TRAIN_STATES)
        valid_spots, valid_running = make_states(day_index, params, VALIDATION_STATES, validation=True)
        train_values, train_stderr = simulate_labels(
            train_spots, train_running, day_index, params, train_paths, params.seed + 10_000 + day_index
        )
        benchmark, benchmark_stderr = simulate_labels(
            valid_spots,
            valid_running,
            day_index,
            params,
            benchmark_paths,
            params.seed + 20_000 + day_index,
        )
        best_method = None
        best_max = float("inf")
        for method in METHODS:
            proxy = fit_proxy(train_spots, train_running, train_values, day_index, params, method)
            prediction = proxy(valid_spots, valid_running)
            metrics = score(prediction, benchmark, benchmark_stderr)
            if metrics["max_rel"] < best_max:
                best_max = metrics["max_rel"]
                best_method = method
                best_predictions[day_index] = prediction
            method_rows.append(
                {
                    "method": method,
                    "day_index": day_index,
                    "remaining_fixings_after_today": future_count(day_index, params),
                    "state_dimension": params.n_assets + 1,
                    "train_states": TRAIN_STATES,
                    "validation_states": VALIDATION_STATES,
                    "train_paths_per_state": train_paths,
                    "benchmark_paths_per_state": benchmark_paths,
                    "avg_train_stderr": float(np.mean(train_stderr)),
                    "avg_benchmark_stderr": float(np.mean(benchmark_stderr)),
                    **metrics,
                }
            )
            for idx in range(len(valid_running)):
                detail_rows.append(
                    {
                        "method": method,
                        "day_index": day_index,
                        "running_sum_before": float(valid_running[idx]),
                        "basket_spot": float(basket_level(valid_spots[idx])),
                        "benchmark": float(benchmark[idx]),
                        "proxy": float(prediction[idx]),
                        "signed_error": float(prediction[idx] - benchmark[idx]),
                        "relative_error": float(abs(prediction[idx] - benchmark[idx]) / max(abs(benchmark[idx]), RELATIVE_ERROR_FLOOR)),
                        "benchmark_stderr": float(benchmark_stderr[idx]),
                    }
                )
        draw_plot(PLOT_DIR / f"basket_asian_day_{day_index:02d}_{best_method}.png", day_index, benchmark, best_predictions[day_index])
        print(f"day {day_index}: best {best_method}, max error {100.0 * best_max:.3f}%")
    write_csv(METHOD_CSV, method_rows)
    write_csv(DETAIL_CSV, detail_rows)
    lines = [
        "# 10-asset basket Asian proxy experiment",
        "",
        "Monthly arithmetic Asian call on an equal-weight basket of 10 correlated GBM underlyings.",
        "The correlation matrix intentionally includes both positive and negative correlations.",
        "",
        "Default method: `pchip_calibrated_log_factor_pca`.",
        "",
        "Training labels use Sobol low-discrepancy paths and boundary-enriched state sampling.",
        "This script does not yet use likelihood-ratio importance sampling for the path simulation measure.",
        f"Training state-scenarios per date are about {TRAIN_STATES * train_paths:,}.",
        f"Benchmark paths per validation state are {qmc_path_count(BENCHMARK_PATHS_PER_STATE):,}.",
        "",
        "| Method | Worst Max % Error | Avg P99 % Error | Avg MAE |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        rows = [row for row in method_rows if row["method"] == method]
        lines.append(
            f"| `{method}` | {100.0 * max(row['max_rel'] for row in rows):.3f}% | "
            f"{100.0 * np.mean([row['p99_rel'] for row in rows]):.3f}% | "
            f"{np.mean([row['mae'] for row in rows]):.6f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"results: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
