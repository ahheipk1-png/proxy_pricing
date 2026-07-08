from math import exp, sqrt
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_stage1_smoothers import (
    ModelFit,
    binned_series,
    bspline_design,
    fit_bspline,
    stream_training_targets,
    weighted_quantile,
    weighted_ridge,
)
from experiment_stage1_accuracy import three_region_values
from plot_stage1_10m_proxy import (
    RELATIVE_ERROR_CAP,
    RELATIVE_ERROR_FLOOR,
    draw_axes,
    draw_metric_panel,
    draw_text,
)
from stage1_lsmc_european import (
    GBMParams,
    asymptotic_anchor_value,
    black_scholes_value,
    option_upper_bound,
)


OUTPUT_PATH = Path("stage1_10m_bspline_tail_anchor_diagnostics.png")
TAIL_DELTA = 0.001
TRAINED_DELTA = 0.01
PLOT_DELTA_LOW = 1e-4
PLOT_DELTA_HIGH = 1.0 - 1e-4
ANCHOR_POINTS_PER_TAIL = 28
ANCHOR_WEIGHT_MULTIPLIER = 0.02


def spot_for_call_delta(delta: float, tau: float, params: GBMParams) -> float:
    if tau <= 0.0:
        return params.strike

    normal = NormalDist()
    d1 = normal.inv_cdf(delta)
    sigma_sqrt_tau = params.vol * sqrt(tau)
    exponent = d1 * sigma_sqrt_tau - (
        params.rate - params.div_yield + 0.5 * params.vol**2
    ) * tau
    return params.strike * exp(exponent)


def wide_spot_grid(
    tau: float, train_spot: np.ndarray, params: GBMParams
) -> np.ndarray:
    low = spot_for_call_delta(PLOT_DELTA_LOW, tau, params)
    high = spot_for_call_delta(PLOT_DELTA_HIGH, tau, params)
    low = min(low, float(train_spot.min()) * 0.8)
    high = max(high, float(train_spot.max()) * 1.12)
    return np.linspace(low, high, 220)


def tail_anchor_spots(tau: float, params: GBMParams) -> np.ndarray:
    if tau <= 0.0:
        return np.array([], dtype=float)

    low_deltas = np.geomspace(
        PLOT_DELTA_LOW, TRAINED_DELTA, ANCHOR_POINTS_PER_TAIL
    )
    high_deltas = 1.0 - low_deltas[::-1]
    deltas = np.concatenate((low_deltas, high_deltas))
    return np.array([spot_for_call_delta(float(delta), tau, params) for delta in deltas])


def fit_tail_anchor_bspline(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    tau: float,
    params: GBMParams,
    n_internal_knots: int,
    ridge: float,
) -> ModelFit:
    if tau <= 0.0:
        return ModelFit(
            name="maturity payoff",
            detail="tau=0",
            predict=lambda new_spot: asymptotic_anchor_value(new_spot, tau, params),
        )

    anchor_spot = tail_anchor_spots(tau, params)
    anchor_target = asymptotic_anchor_value(anchor_spot, tau, params)

    anchor_weight = float(np.median(weights) * ANCHOR_WEIGHT_MULTIPLIER)
    combined_spot = np.concatenate((spot, anchor_spot))
    combined_target = np.concatenate((target, anchor_target))
    combined_weights = np.concatenate(
        (weights, np.full(anchor_spot.size, anchor_weight))
    )

    train_z = spot / params.s0 - 1.0
    combined_z = combined_spot / params.s0 - 1.0
    probabilities = np.linspace(0.0, 1.0, n_internal_knots + 2)[1:-1]
    internal = weighted_quantile(train_z, weights, probabilities)
    z_min = float(combined_z.min())
    z_max = float(combined_z.max())
    knots = np.concatenate(([z_min] * 4, internal, [z_max] * 4))
    design = bspline_design(combined_z, knots)
    coeffs = weighted_ridge(design, combined_target, combined_weights, ridge)

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_spot = np.asarray(new_spot, dtype=float)
        new_z = np.clip(new_spot / params.s0 - 1.0, z_min, z_max)
        raw = bspline_design(new_z, knots) @ coeffs
        return np.minimum(
            np.maximum(raw, asymptotic_anchor_value(new_spot, tau, params)),
            option_upper_bound(new_spot, tau, params),
        )

    return ModelFit(
        name="tail-anchored B-spline",
        detail=(
            f"internal_knots={n_internal_knots}, lambda={ridge:g}, "
            f"anchor_points={anchor_spot.size}, anchor_weight={anchor_weight:.0f}"
        ),
        predict=predict,
    )


def draw_wide_value_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    time: float,
    grid_spot: np.ndarray,
    proxy_values: np.ndarray,
    closed_form_values: np.ndarray,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    x_min = float(grid_spot.min())
    x_max = float(grid_spot.max())
    y_max = float(max(proxy_values.max(), closed_form_values.max(), 1.0) * 1.08)
    sx, sy, _, plot_x1, _, _ = draw_axes(
        draw,
        panel,
        f"t={time:.2f} values",
        "value",
        x_min,
        x_max,
        0.0,
        y_max,
        font,
        small_font,
    )

    proxy_points = [
        (sx(float(x)), sy(float(y))) for x, y in zip(grid_spot, proxy_values)
    ]
    closed_form_points = [
        (sx(float(x)), sy(float(y))) for x, y in zip(grid_spot, closed_form_values)
    ]
    draw.line(proxy_points, fill="#c7352b", width=3)
    draw.line(closed_form_points, fill="#111827", width=2)

    _, y0, _, _ = panel
    draw.line((plot_x1 - 138, y0 + 18, plot_x1 - 100, y0 + 18), fill="#c7352b", width=3)
    draw_text(draw, (plot_x1 - 94, y0 + 11), "proxy", "#111111", small_font)
    draw.line((plot_x1 - 138, y0 + 34, plot_x1 - 100, y0 + 34), fill="#111827", width=2)
    draw_text(draw, (plot_x1 - 94, y0 + 27), "closed form", "#111111", small_font)


def draw_signed_error_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    time: float,
    grid_spot: np.ndarray,
    signed_error: np.ndarray,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    x_min = float(grid_spot.min())
    x_max = float(grid_spot.max())
    y_abs = float(max(np.max(np.abs(signed_error)), 1e-9) * 1.10)
    sx, sy, plot_x0, plot_x1, _, _ = draw_axes(
        draw,
        panel,
        f"t={time:.2f} signed error",
        "proxy-BS",
        x_min,
        x_max,
        -y_abs,
        y_abs,
        font,
        small_font,
        y_formatter=lambda value: f"{value:.2f}",
    )

    zero_y = sy(0.0)
    draw.line((plot_x0, zero_y, plot_x1, zero_y), fill="#444444", width=1)
    points = [(sx(float(x)), sy(float(y))) for x, y in zip(grid_spot, signed_error)]
    for i in range(len(points) - 1):
        midpoint_error = 0.5 * (signed_error[i] + signed_error[i + 1])
        color = "#c7352b" if midpoint_error >= 0.0 else "#2364aa"
        draw.line((points[i], points[i + 1]), fill=color, width=2)

    for x, y in zip(grid_spot, signed_error):
        color = "#c7352b" if y >= 0.0 else "#2364aa"
        x_pix = sx(float(x))
        y_pix = sy(float(y))
        draw.ellipse((x_pix - 2, y_pix - 2, x_pix + 2, y_pix + 2), fill=color)

    _, y0, _, _ = panel
    draw.line((plot_x1 - 144, y0 + 18, plot_x1 - 106, y0 + 18), fill="#c7352b", width=2)
    draw_text(draw, (plot_x1 - 100, y0 + 11), "above BS", "#111111", small_font)
    draw.line((plot_x1 - 144, y0 + 34, plot_x1 - 106, y0 + 34), fill="#2364aa", width=2)
    draw_text(draw, (plot_x1 - 100, y0 + 27), "below BS", "#111111", small_font)


def write_bspline_wide_plot(
    path: Path,
    times: np.ndarray,
    bin_edges_by_time: list[np.ndarray],
    bin_sums_by_time: list[np.ndarray],
    bin_counts_by_time: list[np.ndarray],
    params: GBMParams,
) -> list[tuple[float, float, float, float, float, float, float, int]]:
    width = 2100
    height = 1980
    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    draw_text(draw, (28, 20), "10M-path tail-anchored B-spline diagnostics", "#111111", font)
    draw_text(
        draw,
        (28, 42),
        "Synthetic asymptotic tail points stabilize the spline before the wide plotted tails.",
        "#444444",
        small_font,
    )

    left = 28
    top = 82
    gap_x = 20
    gap_y = 16
    panel_w = (width - 2 * left - 2 * gap_x) // 3
    panel_h = 330
    metrics = []

    for row, step in enumerate(range(1, params.n_steps + 1)):
        y0 = top + row * (panel_h + gap_y)
        panels = []
        for col in range(3):
            x0 = left + col * (panel_w + gap_x)
            panels.append((x0, y0, x0 + panel_w, y0 + panel_h))

        time = float(times[step])
        tau = float(params.maturity - time)
        train_spot, training_ce, counts = binned_series(
            bin_edges_by_time[step], bin_sums_by_time[step], bin_counts_by_time[step]
        )
        weights = counts.astype(float)
        baseline_model = fit_bspline(
            train_spot,
            training_ce,
            weights,
            params,
            n_internal_knots=14,
            ridge=1e-6,
        )
        model = fit_tail_anchor_bspline(
            train_spot,
            training_ce,
            weights,
            tau,
            params,
            n_internal_knots=14,
            ridge=1e-6,
        )

        grid_spot = wide_spot_grid(tau, train_spot, params)
        learned_values = model.predict(grid_spot)
        proxy_values = three_region_values(
            grid_spot,
            tau,
            learned_values,
            params,
            tail_delta=TAIL_DELTA,
            trained_delta=TRAINED_DELTA,
        )
        closed_form_values = black_scholes_value(grid_spot, tau, params)
        error = proxy_values - closed_form_values
        abs_error = np.abs(error)
        rel_error = abs_error / np.maximum(
            np.abs(closed_form_values), RELATIVE_ERROR_FLOOR
        )

        train_closed_form = black_scholes_value(train_spot, tau, params)
        baseline_learned = baseline_model.predict(train_spot)
        baseline_proxy = three_region_values(
            train_spot,
            tau,
            baseline_learned,
            params,
            tail_delta=TAIL_DELTA,
            trained_delta=TRAINED_DELTA,
        )
        baseline_error = baseline_proxy - train_closed_form
        train_learned = model.predict(train_spot)
        train_proxy = three_region_values(
            train_spot,
            tau,
            train_learned,
            params,
            tail_delta=TAIL_DELTA,
            trained_delta=TRAINED_DELTA,
        )
        train_error = train_proxy - train_closed_form
        norm_weights = weights / weights.sum()
        baseline_mae = float(np.sum(norm_weights * np.abs(baseline_error)))
        baseline_rmse = float(np.sqrt(np.sum(norm_weights * baseline_error**2)))
        baseline_rel_mae = float(
            np.sum(
                norm_weights
                * np.abs(baseline_error)
                / np.maximum(np.abs(train_closed_form), RELATIVE_ERROR_FLOOR)
            )
        )
        mae = float(np.sum(norm_weights * np.abs(train_error)))
        rmse = float(np.sqrt(np.sum(norm_weights * train_error**2)))
        rel_mae = float(
            np.sum(
                norm_weights
                * np.abs(train_error)
                / np.maximum(np.abs(train_closed_form), RELATIVE_ERROR_FLOOR)
            )
        )

        draw_wide_value_panel(
            draw,
            panels[0],
            time,
            grid_spot,
            proxy_values,
            closed_form_values,
            font,
            small_font,
        )
        draw_signed_error_panel(
            draw,
            panels[1],
            time,
            grid_spot,
            error,
            font,
            small_font,
        )
        clipped = draw_metric_panel(
            draw,
            panels[2],
            time,
            "rel error",
            "rel err",
            grid_spot,
            rel_error,
            "#6b46c1",
            font,
            small_font,
            y_cap=RELATIVE_ERROR_CAP,
            y_formatter=lambda value: f"{100.0 * value:.0f}%",
        )
        metrics.append(
            (
                time,
                mae,
                rmse,
                rel_mae,
                baseline_mae,
                baseline_rmse,
                baseline_rel_mae,
                clipped,
            )
        )

    summary_y = top + params.n_steps * panel_h + (params.n_steps - 1) * gap_y + 24
    draw_text(draw, (28, summary_y), "summary", "#111111", font)
    draw_text(draw, (28, summary_y + 28), f"training paths: {params.n_paths:,}", "#333333", small_font)
    draw_text(draw, (28, summary_y + 50), "model: B-spline, direct asymptotic tail anchors, 14 internal knots, lambda=1e-6", "#333333", small_font)
    draw_text(draw, (28, summary_y + 72), f"asymptotic mix: tail={TAIL_DELTA:g}, trained={TRAINED_DELTA:g}", "#333333", small_font)
    draw_text(
        draw,
        (28, summary_y + 94),
        f"anchors: {ANCHOR_POINTS_PER_TAIL} per tail, weight={ANCHOR_WEIGHT_MULTIPLIER:g}x median bin count",
        "#333333",
        small_font,
    )
    draw_text(
        draw,
        (450, summary_y + 28),
        "summary metrics are weighted on the simulated training spot range; plotted curves use a wider delta-based spot range.",
        "#333333",
        small_font,
    )
    for row, (
        time,
        mae,
        rmse,
        rel_mae,
        baseline_mae,
        _baseline_rmse,
        _baseline_rel_mae,
        clipped,
    ) in enumerate(metrics):
        draw_text(
            draw,
            (450, summary_y + 56 + row * 22),
            f"t={time:.2f}: anchored MAE={mae:.5f}, baseline MAE={baseline_mae:.5f}, rel_MAE={100.0 * rel_mae:.2f}%, plotted clipped_bins={clipped}",
            "#333333",
            small_font,
        )

    image.save(path)
    return metrics


def main() -> None:
    params = GBMParams(n_paths=10_000_000, n_steps=5, seed=7, option_type="call")
    start = perf_counter()
    times, _path_coeffs, bin_edges_by_time, bin_sums_by_time, bin_counts_by_time = (
        stream_training_targets(params)
    )
    metrics = write_bspline_wide_plot(
        OUTPUT_PATH,
        times,
        bin_edges_by_time,
        bin_sums_by_time,
        bin_counts_by_time,
        params,
    )

    print()
    print("10M-path tail-anchored B-spline diagnostics")
    print(f"plot delta range: {PLOT_DELTA_LOW:g} to {PLOT_DELTA_HIGH:g}")
    print(
        f"tail anchors: {ANCHOR_POINTS_PER_TAIL} per tail, "
        f"weight={ANCHOR_WEIGHT_MULTIPLIER:g}x median bin count"
    )
    print("summary metrics are weighted on the simulated training spot range")
    print("time    anchored MAE  baseline MAE  anchored RMSE  baseline RMSE  anchored rel")
    print("----    ------------  ------------  -------------  -------------  ------------")
    for (
        time,
        mae,
        rmse,
        rel_mae,
        baseline_mae,
        baseline_rmse,
        _baseline_rel_mae,
        _clipped,
    ) in metrics:
        print(
            f"{time:4.2f}    {mae:12.5f}  {baseline_mae:12.5f}  "
            f"{rmse:13.5f}  {baseline_rmse:13.5f}  {100.0 * rel_mae:11.2f}%"
        )
    print()
    print(f"plot written to: {OUTPUT_PATH.resolve()}")
    print(f"elapsed seconds: {perf_counter() - start:.1f}")


if __name__ == "__main__":
    main()
