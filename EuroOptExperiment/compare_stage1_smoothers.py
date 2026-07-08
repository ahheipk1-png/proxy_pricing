from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stage1_lsmc_european import (
    GBMParams,
    black_scholes_value,
    european_payoff,
    polynomial_basis,
    proxy_value,
)


OUTPUT_PATH = Path("stage1_10m_method_bakeoff.png")
CSV_PATH = Path("stage1_10m_method_bakeoff.csv")
N_BINS = 140
CHUNK_SIZE = 500_000
RELATIVE_ERROR_FLOOR = 0.01


@dataclass(frozen=True)
class ModelFit:
    name: str
    detail: str
    predict: object


@dataclass(frozen=True)
class Score:
    method: str
    detail: str
    time: float
    mae: float
    rmse: float
    rel_mae: float
    max_abs_error: float


def lognormal_spot_edges(time: float, params: GBMParams, n_bins: int) -> np.ndarray:
    if time <= 0.0:
        return np.linspace(params.s0 * 0.999, params.s0 * 1.001, n_bins + 1)

    normal = NormalDist()
    q_low = normal.inv_cdf(0.005)
    q_high = normal.inv_cdf(0.995)
    mean = (params.rate - params.div_yield - 0.5 * params.vol**2) * time
    stdev = params.vol * sqrt(time)
    low = params.s0 * exp(mean + stdev * q_low)
    high = params.s0 * exp(mean + stdev * q_high)
    return np.linspace(low, high, n_bins + 1)


def stream_training_targets(
    params: GBMParams, n_bins: int = N_BINS, chunk_size: int = CHUNK_SIZE
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(params.seed)
    dt = params.maturity / params.n_steps
    times = np.linspace(0.0, params.maturity, params.n_steps + 1)
    discounts = np.exp(-params.rate * (params.maturity - times))
    drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
    diffusion = params.vol * sqrt(dt)

    n_basis = polynomial_basis(np.array([params.s0]), params).shape[1]
    xtx_by_time = [np.zeros((n_basis, n_basis)) for _ in times]
    xty_by_time = [np.zeros(n_basis) for _ in times]
    path_poly_coeffs_by_time = [np.zeros(n_basis) for _ in times]
    target_sum_at_start = 0.0

    bin_edges_by_time = [
        lognormal_spot_edges(float(time), params, n_bins) for time in times
    ]
    bin_sums_by_time = [np.zeros(n_bins) for _ in times]
    bin_counts_by_time = [np.zeros(n_bins, dtype=np.int64) for _ in times]

    processed = 0
    while processed < params.n_paths:
        n_chunk = min(chunk_size, params.n_paths - processed)
        paths = np.empty((n_chunk, params.n_steps + 1), dtype=np.float64)
        paths[:, 0] = params.s0

        shocks = rng.standard_normal((n_chunk, params.n_steps))
        for step in range(params.n_steps):
            paths[:, step + 1] = paths[:, step] * np.exp(
                drift + diffusion * shocks[:, step]
            )

        terminal_payoff = european_payoff(paths[:, -1], params)
        target_sum_at_start += float((discounts[0] * terminal_payoff).sum())

        for step in range(1, params.n_steps + 1):
            target = discounts[step] * terminal_payoff
            spot = paths[:, step]

            basis = polynomial_basis(spot, params)
            xtx_by_time[step] += basis.T @ basis
            xty_by_time[step] += basis.T @ target

            edges = bin_edges_by_time[step]
            bin_index = np.searchsorted(edges, spot, side="right") - 1
            valid = (bin_index >= 0) & (bin_index < n_bins)
            valid_bin_index = bin_index[valid]
            bin_counts_by_time[step] += np.bincount(
                valid_bin_index, minlength=n_bins
            )
            bin_sums_by_time[step] += np.bincount(
                valid_bin_index, weights=target[valid], minlength=n_bins
            )

        processed += n_chunk
        print(f"processed {processed:,} / {params.n_paths:,} paths")

    path_poly_coeffs_by_time[0][0] = target_sum_at_start / params.n_paths
    for step in range(1, params.n_steps + 1):
        path_poly_coeffs_by_time[step] = (
            np.linalg.pinv(xtx_by_time[step]) @ xty_by_time[step]
        )

    return (
        times,
        path_poly_coeffs_by_time,
        bin_edges_by_time,
        bin_sums_by_time,
        bin_counts_by_time,
    )


def binned_series(
    bin_edges: np.ndarray, bin_sums: np.ndarray, bin_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = bin_counts > 0
    return centers[valid], bin_sums[valid] / bin_counts[valid], bin_counts[valid]


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    return np.interp(probabilities, cumulative, values)


def weighted_ridge(
    x_matrix: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge: float,
    unpenalized_columns: int = 1,
) -> np.ndarray:
    weighted_x = x_matrix * np.sqrt(weights)[:, None]
    weighted_y = y * np.sqrt(weights)
    xtx = weighted_x.T @ weighted_x
    penalty = np.eye(x_matrix.shape[1]) * ridge
    penalty[:unpenalized_columns, :unpenalized_columns] = 0.0
    xty = weighted_x.T @ weighted_y
    return np.linalg.solve(xtx + penalty, xty)


def polynomial_design(spot: np.ndarray, params: GBMParams, degree: int) -> np.ndarray:
    x = np.asarray(spot, dtype=float) / params.s0 - 1.0
    columns = [x**power for power in range(degree + 1)]
    columns.append(european_payoff(np.asarray(spot, dtype=float), params) / params.s0)
    return np.column_stack(columns)


def fit_weighted_polynomial(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    params: GBMParams,
    degree: int,
    ridge: float,
) -> ModelFit:
    design = polynomial_design(spot, params, degree)
    coeffs = weighted_ridge(design, target, weights, ridge)
    return ModelFit(
        name="ridge polynomial",
        detail=f"degree={degree}, lambda={ridge:g}",
        predict=lambda new_spot: polynomial_design(new_spot, params, degree) @ coeffs,
    )


def bspline_design(
    z: np.ndarray, knots: np.ndarray, degree: int = 3
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    n_basis = len(knots) - degree - 1
    basis = np.zeros((z.size, n_basis + degree))

    for i in range(n_basis + degree):
        left = knots[i]
        right = knots[i + 1]
        basis[:, i] = ((z >= left) & (z < right)).astype(float)
    basis[z == knots[-1], n_basis - 1] = 1.0

    for current_degree in range(1, degree + 1):
        next_basis = np.zeros((z.size, n_basis + degree - current_degree))
        for i in range(n_basis + degree - current_degree):
            left_denominator = knots[i + current_degree] - knots[i]
            right_denominator = knots[i + current_degree + 1] - knots[i + 1]
            if left_denominator > 0.0:
                next_basis[:, i] += (
                    (z - knots[i]) / left_denominator * basis[:, i]
                )
            if right_denominator > 0.0:
                next_basis[:, i] += (
                    (knots[i + current_degree + 1] - z)
                    / right_denominator
                    * basis[:, i + 1]
                )
        basis = next_basis

    return basis[:, :n_basis]


def fit_bspline(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    params: GBMParams,
    n_internal_knots: int,
    ridge: float,
) -> ModelFit:
    z = spot / params.s0 - 1.0
    probabilities = np.linspace(0.0, 1.0, n_internal_knots + 2)[1:-1]
    internal = weighted_quantile(z, weights, probabilities)
    z_min = float(z.min())
    z_max = float(z.max())
    knots = np.concatenate(([z_min] * 4, internal, [z_max] * 4))
    design = bspline_design(z, knots)
    coeffs = weighted_ridge(design, target, weights, ridge)

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_z = np.clip(np.asarray(new_spot, dtype=float) / params.s0 - 1.0, z_min, z_max)
        return bspline_design(new_z, knots) @ coeffs

    return ModelFit(
        name="ridge B-spline",
        detail=f"internal_knots={n_internal_knots}, lambda={ridge:g}",
        predict=predict,
    )


def natural_cubic_design(z: np.ndarray, knots: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    basis = [np.ones_like(z), z]
    last_knot = knots[-1]
    second_last_knot = knots[-2]
    denominator_last = last_knot - second_last_knot

    def truncated_cube(value: np.ndarray, knot: float) -> np.ndarray:
        return np.maximum(value - knot, 0.0) ** 3

    for knot in knots[1:-2]:
        first = (
            truncated_cube(z, knot) - truncated_cube(z, last_knot)
        ) / (last_knot - knot)
        second = (
            truncated_cube(z, second_last_knot) - truncated_cube(z, last_knot)
        ) / denominator_last
        basis.append(first - second)

    return np.column_stack(basis)


def fit_natural_cubic(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    params: GBMParams,
    n_internal_knots: int,
    ridge: float,
) -> ModelFit:
    z = spot / params.s0 - 1.0
    probabilities = np.linspace(0.0, 1.0, n_internal_knots + 2)
    knots = weighted_quantile(z, weights, probabilities)
    z_min = float(knots[0])
    z_max = float(knots[-1])
    design = natural_cubic_design(z, knots)
    coeffs = weighted_ridge(design, target, weights, ridge, unpenalized_columns=2)

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_z = np.clip(np.asarray(new_spot, dtype=float) / params.s0 - 1.0, z_min, z_max)
        return natural_cubic_design(new_z, knots) @ coeffs

    return ModelFit(
        name="natural cubic spline",
        detail=f"internal_knots={n_internal_knots}, lambda={ridge:g}",
        predict=predict,
    )


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)

    for i in range(1, len(y) - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            slopes[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    slopes[0] = delta[0]
    slopes[-1] = delta[-1]
    return slopes


def fit_pchip(spot: np.ndarray, target: np.ndarray) -> ModelFit:
    x = np.asarray(spot, dtype=float)
    y = np.asarray(target, dtype=float)
    slopes = pchip_slopes(x, y)

    def predict(new_spot: np.ndarray) -> np.ndarray:
        values = np.asarray(new_spot, dtype=float)
        indices = np.searchsorted(x, values, side="right") - 1
        indices = np.clip(indices, 0, len(x) - 2)
        h = x[indices + 1] - x[indices]
        t = (values - x[indices]) / h
        t = np.clip(t, 0.0, 1.0)
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        return (
            h00 * y[indices]
            + h10 * h * slopes[indices]
            + h01 * y[indices + 1]
            + h11 * h * slopes[indices + 1]
        )

    return ModelFit(name="PCHIP", detail="shape-preserving cubic", predict=predict)


def loess_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    query_x: np.ndarray,
    frac: float,
    degree: int = 2,
) -> np.ndarray:
    k = max(degree + 2, int(np.ceil(frac * len(train_x))))
    output = np.empty_like(query_x, dtype=float)

    for idx, value in enumerate(query_x):
        distances = np.abs(train_x - value)
        nearest = np.argpartition(distances, k - 1)[:k]
        scale = distances[nearest].max()
        if scale <= 0.0:
            output[idx] = train_y[nearest].mean()
            continue
        u = distances[nearest] / scale
        local_weights = (1.0 - u**3) ** 3 * train_weights[nearest]
        centered = train_x[nearest] - value
        if degree == 1:
            design = np.column_stack([np.ones_like(centered), centered])
        else:
            design = np.column_stack([np.ones_like(centered), centered, centered**2])
        coeffs = weighted_ridge(design, train_y[nearest], local_weights, 1e-10)
        output[idx] = coeffs[0]

    return output


def fit_loess(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    frac: float,
    degree: int = 2,
) -> ModelFit:
    return ModelFit(
        name="LOESS",
        detail=f"frac={frac:g}, local_degree={degree}",
        predict=lambda new_spot: loess_predict(
            spot, target, weights, np.asarray(new_spot, dtype=float), frac, degree
        ),
    )


def fit_kernel_smoother(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
    detail: str,
) -> ModelFit:
    def predict(new_spot: np.ndarray) -> np.ndarray:
        values = np.asarray(new_spot, dtype=float)
        out = np.empty_like(values)
        for idx, value in enumerate(values):
            kernel = np.exp(-0.5 * ((spot - value) / bandwidth) ** 2) * weights
            out[idx] = np.sum(kernel * target) / np.sum(kernel)
        return out

    return ModelFit(
        name="Gaussian kernel",
        detail=detail,
        predict=predict,
    )


def clipped_option_values(values: np.ndarray, spot: np.ndarray, params: GBMParams) -> np.ndarray:
    lower = european_payoff(spot, params)
    upper = np.full_like(spot, params.strike)
    if params.option_type == "call":
        upper = spot
    return np.minimum(np.maximum(values, lower), upper)


def score_fit(
    model: ModelFit,
    spot: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    params: GBMParams,
    time: float,
) -> tuple[Score, np.ndarray]:
    raw_prediction = np.asarray(model.predict(spot), dtype=float)
    prediction = clipped_option_values(raw_prediction, spot, params)
    error = prediction - truth
    abs_error = np.abs(error)
    rel_error = abs_error / np.maximum(np.abs(truth), RELATIVE_ERROR_FLOOR)
    normalized_weights = weights / weights.sum()
    return (
        Score(
            method=model.name,
            detail=model.detail,
            time=time,
            mae=float(np.sum(normalized_weights * abs_error)),
            rmse=float(np.sqrt(np.sum(normalized_weights * error**2))),
            rel_mae=float(np.sum(normalized_weights * rel_error)),
            max_abs_error=float(abs_error.max()),
        ),
        prediction,
    )


def build_models_for_time(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    params: GBMParams,
    path_poly_coeffs: np.ndarray,
) -> list[ModelFit]:
    models = [
        ModelFit(
            name="path polynomial",
            detail=f"degree={params.basis_degree}+payoff, path LS",
            predict=lambda new_spot, coeffs=path_poly_coeffs: proxy_value(
                new_spot, coeffs, params
            ),
        )
    ]

    models.append(
        fit_weighted_polynomial(
            spot, target, weights, params, degree=5, ridge=1e-8
        )
    )
    models.append(
        fit_weighted_polynomial(
            spot, target, weights, params, degree=9, ridge=1e-6
        )
    )

    for n_knots, ridge in [(8, 1e-8), (14, 1e-6)]:
        models.append(fit_bspline(spot, target, weights, params, n_knots, ridge))

    for n_knots, ridge in [(6, 1e-8), (10, 1e-6)]:
        models.append(
            fit_natural_cubic(spot, target, weights, params, n_knots, ridge)
        )

    models.append(fit_pchip(spot, target))
    models.append(fit_loess(spot, target, weights, frac=0.22, degree=2))
    models.append(fit_loess(spot, target, weights, frac=0.35, degree=2))

    bandwidth = 0.08 * (spot.max() - spot.min())
    models.append(
        fit_kernel_smoother(
            spot, target, weights, bandwidth, detail="bandwidth=8% spot range"
        )
    )
    return models


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.text(xy, text, fill=fill, font=font)


def draw_leaderboard(
    path: Path,
    times: np.ndarray,
    plots_by_step: list[dict[str, object]],
    overall_scores: list[Score],
) -> None:
    width = 2200
    height = 1780
    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    colors = {
        "path polynomial": "#c7352b",
        "ridge polynomial": "#d97706",
        "ridge B-spline": "#2364aa",
        "natural cubic spline": "#2f855a",
        "PCHIP": "#6b46c1",
        "LOESS": "#111827",
        "Gaussian kernel": "#64748b",
    }

    draw_text(draw, (28, 20), "10M-path smoother bake-off", "#111111", font)
    draw_text(
        draw,
        (28, 42),
        "Models train on simulated binned conditional expectations and are scored against Black-Scholes on the same spot distribution.",
        "#444444",
        small_font,
    )

    left = 28
    top = 82
    gap_y = 18
    panel_w = 1370
    panel_h = 290
    table_x = 1430

    for row, plot in enumerate(plots_by_step):
        y0 = top + row * (panel_h + gap_y)
        x0 = left
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.rectangle((x0, y0, x1, y1), fill="#ffffff")
        draw.rectangle((x0 + 58, y0 + 36, x1 - 20, y1 - 38), outline="#333333")

        spot = plot["spot"]
        truth = plot["truth"]
        abs_errors = plot["abs_errors"]
        time = plot["time"]
        best_methods = plot["best_methods"]
        x_min = float(spot.min())
        x_max = float(spot.max())
        y_max = max(float(np.max(abs_errors[method])) for method in best_methods) * 1.08
        y_max = max(y_max, 0.01)

        def sx(value: float) -> int:
            return int((x0 + 58) + (value - x_min) / (x_max - x_min) * (panel_w - 78))

        def sy(value: float) -> int:
            return int((y1 - 38) - value / y_max * (panel_h - 74))

        for tick in range(5):
            x_tick = x_min + (x_max - x_min) * tick / 4
            x_pix = sx(x_tick)
            draw.line((x_pix, y0 + 36, x_pix, y1 - 38), fill="#eeeeee")
            draw_text(draw, (x_pix - 16, y1 - 28), f"{x_tick:.0f}", "#444444", small_font)
            y_tick = y_max * tick / 4
            y_pix = sy(y_tick)
            draw.line((x0 + 58, y_pix, x1 - 20, y_pix), fill="#eeeeee")
            draw_text(draw, (x0 + 8, y_pix - 6), f"{y_tick:.2f}", "#444444", small_font)

        draw_text(draw, (x0 + 12, y0 + 8), f"t={time:.2f} abs error vs BS", "#111111", font)
        draw_text(draw, (x0 + 8, y0 + 24), "|error|", "#444444", small_font)
        draw_text(draw, (x0 + 58, y1 - 18), "spot", "#444444", small_font)

        for method in best_methods:
            values = abs_errors[method]
            points = [(sx(float(x)), sy(float(y))) for x, y in zip(spot, values)]
            draw.line(points, fill=colors.get(method.split(" [")[0], "#111111"), width=2)

        legend_x = x1 - 310
        for idx, method in enumerate(best_methods):
            legend_y = y0 + 14 + idx * 18
            base_name = method.split(" [")[0]
            draw.line(
                (legend_x, legend_y + 6, legend_x + 28, legend_y + 6),
                fill=colors.get(base_name, "#111111"),
                width=2,
            )
            draw_text(draw, (legend_x + 34, legend_y), method, "#111111", small_font)

    draw.rectangle((table_x, top, width - 28, top + 760), fill="#ffffff")
    draw.rectangle((table_x, top, width - 28, top + 760), outline="#333333")
    draw_text(draw, (table_x + 20, top + 20), "overall ranking", "#111111", font)
    draw_text(
        draw,
        (table_x + 20, top + 45),
        "weighted across all displayed times",
        "#444444",
        small_font,
    )
    draw_text(
        draw,
        (table_x + 20, top + 82),
        "rank  method                              MAE      RMSE    rel MAE",
        "#111111",
        small_font,
    )

    for rank, score in enumerate(overall_scores[:18], start=1):
        method = f"{score.method} ({score.detail})"
        if len(method) > 34:
            method = method[:31] + "..."
        line = (
            f"{rank:>2}    {method:<34} "
            f"{score.mae:7.5f}  {score.rmse:7.5f}  {100.0 * score.rel_mae:7.2f}%"
        )
        draw_text(draw, (table_x + 20, top + 92 + rank * 30), line, "#333333", small_font)

    image.save(path)


def main() -> None:
    params = GBMParams(n_paths=10_000_000, n_steps=5, seed=7, option_type="call")
    start = perf_counter()
    (
        times,
        path_poly_coeffs_by_time,
        bin_edges_by_time,
        bin_sums_by_time,
        bin_counts_by_time,
    ) = stream_training_targets(params)

    scores: list[Score] = []
    plots_by_step = []

    for step in range(1, params.n_steps + 1):
        spot, target, counts = binned_series(
            bin_edges_by_time[step], bin_sums_by_time[step], bin_counts_by_time[step]
        )
        weights = counts.astype(float)
        tau = params.maturity - times[step]
        truth = black_scholes_value(spot, float(tau), params)
        models = build_models_for_time(
            spot, target, weights, params, path_poly_coeffs_by_time[step]
        )

        predictions: dict[str, np.ndarray] = {}
        abs_errors: dict[str, np.ndarray] = {}
        step_scores = []
        for model in models:
            score, prediction = score_fit(
                model, spot, truth, weights, params, float(times[step])
            )
            label = f"{score.method} [{score.detail}]"
            predictions[label] = prediction
            abs_errors[label] = np.abs(prediction - truth)
            scores.append(score)
            step_scores.append(score)

        best_labels = [
            f"{score.method} [{score.detail}]"
            for score in sorted(step_scores, key=lambda item: item.mae)[:5]
        ]
        plots_by_step.append(
            {
                "time": float(times[step]),
                "spot": spot,
                "truth": truth,
                "abs_errors": abs_errors,
                "best_methods": best_labels,
            }
        )

    method_keys = sorted({(score.method, score.detail) for score in scores})
    overall_scores = []
    for method, detail in method_keys:
        method_scores = [
            score for score in scores if score.method == method and score.detail == detail
        ]
        overall_scores.append(
            Score(
                method=method,
                detail=detail,
                time=-1.0,
                mae=float(np.mean([score.mae for score in method_scores])),
                rmse=float(np.mean([score.rmse for score in method_scores])),
                rel_mae=float(np.mean([score.rel_mae for score in method_scores])),
                max_abs_error=float(max(score.max_abs_error for score in method_scores)),
            )
        )
    overall_scores.sort(key=lambda item: item.mae)

    with CSV_PATH.open("w", encoding="utf-8") as file:
        file.write("method,detail,time,mae,rmse,rel_mae,max_abs_error\n")
        for score in sorted(scores, key=lambda item: (item.time, item.mae)):
            file.write(
                f"{score.method},{score.detail},{score.time:.6f},"
                f"{score.mae:.10f},{score.rmse:.10f},"
                f"{score.rel_mae:.10f},{score.max_abs_error:.10f}\n"
            )
        file.write("\n")
        file.write("overall_method,detail,mae,rmse,rel_mae,max_abs_error\n")
        for score in overall_scores:
            file.write(
                f"{score.method},{score.detail},{score.mae:.10f},"
                f"{score.rmse:.10f},{score.rel_mae:.10f},"
                f"{score.max_abs_error:.10f}\n"
            )

    draw_leaderboard(OUTPUT_PATH, times, plots_by_step, overall_scores)

    print()
    print("10M-path smoother bake-off")
    print("fixed params: European call, S0=100, K=100, r=5%, q=2%, vol=20%, T=1")
    print("scoring target: Black-Scholes conditional value on weighted spot bins")
    print()
    print("overall ranking by average weighted MAE")
    print("rank  method                    detail                              MAE       RMSE      rel MAE")
    print("----  ------                    ------                              ---       ----      -------")
    for rank, score in enumerate(overall_scores, start=1):
        print(
            f"{rank:4d}  {score.method:<24} {score.detail:<34} "
            f"{score.mae:8.5f}  {score.rmse:8.5f}  {100.0 * score.rel_mae:8.2f}%"
        )

    print()
    print(f"plot written to: {OUTPUT_PATH.resolve()}")
    print(f"table written to: {CSV_PATH.resolve()}")
    print(f"elapsed seconds: {perf_counter() - start:.1f}")


if __name__ == "__main__":
    main()
