from math import exp, sqrt
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stage1_lsmc_european import (
    GBMParams,
    black_scholes_value,
    default_proxy_value,
    european_payoff,
    polynomial_basis,
    proxy_model_description,
    ridge_coefficients_from_moments,
)


OUTPUT_PATH = Path("stage1_10m_proxy_diagnostics.png")
N_BINS = 90
CHUNK_SIZE = 500_000
RELATIVE_ERROR_FLOOR = 0.01
RELATIVE_ERROR_CAP = 2.0


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


def fit_streaming_proxy(
    params: GBMParams, chunk_size: int = CHUNK_SIZE, n_bins: int = N_BINS
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
    coeffs_by_time = [np.zeros(n_basis) for _ in times]
    target_sum_at_start = 0.0

    bin_edges_by_time = [
        lognormal_spot_edges(float(time), params, n_bins) for time in times
    ]
    bin_sums_by_time = [np.zeros(n_bins) for _ in times]
    bin_counts_by_time = [np.zeros(n_bins, dtype=np.int64) for _ in times]

    processed = 0
    chunk_number = 0
    while processed < params.n_paths:
        chunk_number += 1
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

    coeffs_by_time[0][0] = target_sum_at_start / params.n_paths

    for step in range(1, params.n_steps + 1):
        if step == params.n_steps:
            coeffs_by_time[step][-1] = params.s0
        else:
            coeffs_by_time[step] = ridge_coefficients_from_moments(
                xtx_by_time[step], xty_by_time[step], params.ridge_lambda
            )

    return times, coeffs_by_time, bin_edges_by_time, bin_sums_by_time, bin_counts_by_time


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.text(xy, text, fill=fill, font=font)


def binned_fit_series(
    params: GBMParams,
    coeffs: np.ndarray,
    tau: float,
    bin_edges: np.ndarray,
    bin_sums: np.ndarray,
    bin_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = bin_counts > 0
    x_values = centers[valid]
    ce_values = bin_sums[valid] / bin_counts[valid]
    proxy_values = default_proxy_value(x_values, tau, coeffs, params)
    counts = bin_counts[valid]
    error = proxy_values - ce_values
    abs_error = np.abs(error)
    rel_error = abs_error / np.maximum(np.abs(ce_values), RELATIVE_ERROR_FLOOR)
    return x_values, ce_values, proxy_values, counts, error, abs_error, rel_error


def draw_axes(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    title: str,
    y_label: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    y_formatter=None,
):
    if y_formatter is None:
        y_formatter = lambda value: f"{value:.1f}"

    x0, y0, x1, y1 = panel
    margin_left = 56
    margin_right = 18
    margin_top = 38
    margin_bottom = 38
    plot_x0 = x0 + margin_left
    plot_x1 = x1 - margin_right
    plot_y0 = y0 + margin_top
    plot_y1 = y1 - margin_bottom

    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(value: float) -> int:
        return int(plot_x0 + (value - x_min) / (x_max - x_min) * (plot_x1 - plot_x0))

    def sy(value: float) -> int:
        return int(plot_y1 - (value - y_min) / (y_max - y_min) * (plot_y1 - plot_y0))

    draw.rectangle(panel, fill="#ffffff")
    draw.rectangle((plot_x0, plot_y0, plot_x1, plot_y1), outline="#333333", width=1)

    for tick in range(5):
        x_tick = x_min + (x_max - x_min) * tick / 4
        x_pix = sx(x_tick)
        draw.line((x_pix, plot_y0, x_pix, plot_y1), fill="#eeeeee")
        draw_text(draw, (x_pix - 18, plot_y1 + 8), f"{x_tick:.0f}", "#444444", small_font)

        y_tick = y_min + (y_max - y_min) * tick / 4
        y_pix = sy(y_tick)
        draw.line((plot_x0, y_pix, plot_x1, y_pix), fill="#eeeeee")
        draw_text(draw, (x0 + 8, y_pix - 6), y_formatter(y_tick), "#444444", small_font)

    draw_text(draw, (x0 + 12, y0 + 8), title, "#111111", font)
    draw_text(draw, (plot_x0, y1 - 24), "spot", "#444444", small_font)
    draw_text(draw, (x0 + 8, y0 + 24), y_label, "#444444", small_font)

    return sx, sy, plot_x0, plot_x1, plot_y0, plot_y1


def draw_value_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    time: float,
    x_values: np.ndarray,
    ce_values: np.ndarray,
    proxy_values: np.ndarray,
    closed_form_values: np.ndarray,
    counts: np.ndarray,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    y_max = float(max(ce_values.max(), proxy_values.max(), closed_form_values.max(), 1.0) * 1.08)
    sx, sy, plot_x0, plot_x1, _, _ = draw_axes(
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

    proxy_points = [(sx(float(x)), sy(float(y))) for x, y in zip(x_values, proxy_values)]
    if len(proxy_points) > 1:
        draw.line(proxy_points, fill="#c7352b", width=3)

    closed_form_points = [
        (sx(float(x)), sy(float(y))) for x, y in zip(x_values, closed_form_values)
    ]
    if len(closed_form_points) > 1:
        draw.line(closed_form_points, fill="#111827", width=2)

    for x, y, count in zip(x_values, ce_values, counts):
        radius = 2 if count < 10_000 else 3
        x_pix = sx(float(x))
        y_pix = sy(float(y))
        draw.ellipse(
            (x_pix - radius, y_pix - radius, x_pix + radius, y_pix + radius),
            fill="#2364aa",
        )

    x0, y0, _, _ = panel
    draw.line((plot_x1 - 138, y0 + 18, plot_x1 - 100, y0 + 18), fill="#c7352b", width=3)
    draw_text(draw, (plot_x1 - 94, y0 + 11), "proxy", "#111111", small_font)
    draw.line((plot_x1 - 138, y0 + 34, plot_x1 - 100, y0 + 34), fill="#111827", width=2)
    draw_text(draw, (plot_x1 - 94, y0 + 27), "closed form", "#111111", small_font)
    draw.ellipse((plot_x1 - 138, y0 + 49, plot_x1 - 132, y0 + 55), fill="#2364aa")
    draw_text(draw, (plot_x1 - 126, y0 + 43), "training CE", "#111111", small_font)


def draw_metric_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    time: float,
    title_suffix: str,
    y_label: str,
    x_values: np.ndarray,
    metric_values: np.ndarray,
    color: str,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    y_cap: float | None = None,
    y_formatter=None,
) -> int:
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    plot_values = metric_values
    clipped = np.zeros_like(metric_values, dtype=bool)
    if y_cap is not None:
        clipped = metric_values > y_cap
        plot_values = np.minimum(metric_values, y_cap)
        y_max = y_cap
    else:
        y_max = float(max(metric_values.max(), 1e-9) * 1.10)

    sx, sy, _, _, _, _ = draw_axes(
        draw,
        panel,
        f"t={time:.2f} {title_suffix}",
        y_label,
        x_min,
        x_max,
        0.0,
        y_max,
        font,
        small_font,
        y_formatter=y_formatter,
    )

    points = [(sx(float(x)), sy(float(y))) for x, y in zip(x_values, plot_values)]
    if len(points) > 1:
        draw.line(points, fill=color, width=2)

    for x, y, is_clipped in zip(x_values, plot_values, clipped):
        x_pix = sx(float(x))
        y_pix = sy(float(y))
        point_color = "#8a2d2d" if is_clipped else color
        draw.ellipse((x_pix - 2, y_pix - 2, x_pix + 2, y_pix + 2), fill=point_color)

    if np.any(clipped):
        _, y0, x1, _ = panel
        draw_text(
            draw,
            (x1 - 120, y0 + 10),
            f"{int(np.count_nonzero(clipped))} bins clipped",
            "#8a2d2d",
            small_font,
        )

    return int(np.count_nonzero(clipped))


def write_plot(
    path: Path,
    times: np.ndarray,
    coeffs_by_time: list[np.ndarray],
    bin_edges_by_time: list[np.ndarray],
    bin_sums_by_time: list[np.ndarray],
    bin_counts_by_time: list[np.ndarray],
    params: GBMParams,
) -> list[tuple[float, float, float, float, int]]:
    width = 2100
    height = 1980
    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    draw_text(
        draw,
        (28, 20),
        "10M-path European payoff proxy diagnostics",
        "#111111",
        font,
    )
    draw_text(
        draw,
        (28, 42),
        "Each row is a training time. Values show proxy, closed-form Black-Scholes, and binned training CE; errors compare proxy to closed form.",
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

        (
            x_values,
            ce_values,
            proxy_values,
            counts,
            _training_error,
            _training_abs_error,
            _training_rel_error,
        ) = binned_fit_series(
            params,
            coeffs_by_time[step],
            float(params.maturity - times[step]),
            bin_edges_by_time[step],
            bin_sums_by_time[step],
            bin_counts_by_time[step],
        )
        tau = params.maturity - times[step]
        closed_form_values = black_scholes_value(x_values, float(tau), params)
        error = proxy_values - closed_form_values
        abs_error = np.abs(error)
        rel_error = abs_error / np.maximum(np.abs(closed_form_values), RELATIVE_ERROR_FLOOR)
        weights = counts / counts.sum()
        mae = float(np.sum(weights * abs_error))
        rmse = float(np.sqrt(np.sum(weights * error**2)))
        rel_mae = float(np.sum(weights * rel_error))

        draw_value_panel(
            draw,
            panels[0],
            float(times[step]),
            x_values,
            ce_values,
            proxy_values,
            closed_form_values,
            counts,
            font,
            small_font,
        )
        draw_metric_panel(
            draw,
            panels[1],
            float(times[step]),
            "abs error",
            "|proxy-CE|",
            x_values,
            abs_error,
            "#2f855a",
            font,
            small_font,
            y_formatter=lambda value: f"{value:.2f}",
        )
        clipped = draw_metric_panel(
            draw,
            panels[2],
            float(times[step]),
            "rel error",
            "rel err",
            x_values,
            rel_error,
            "#6b46c1",
            font,
            small_font,
            y_cap=RELATIVE_ERROR_CAP,
            y_formatter=lambda value: f"{100.0 * value:.0f}%",
        )
        metrics.append((float(times[step]), mae, rmse, rel_mae, clipped))

    summary_y = top + params.n_steps * panel_h + (params.n_steps - 1) * gap_y + 24
    draw_text(draw, (28, summary_y), "summary", "#111111", font)
    initial_proxy = coeffs_by_time[0][0]
    initial_bs = float(black_scholes_value(np.array([params.s0]), params.maturity, params)[0])
    draw_text(draw, (28, summary_y + 28), f"training paths: {params.n_paths:,}", "#333333", small_font)
    draw_text(draw, (28, summary_y + 50), f"initial proxy: {initial_proxy:.6f}", "#333333", small_font)
    draw_text(draw, (28, summary_y + 72), f"Black-Scholes: {initial_bs:.6f}", "#333333", small_font)
    draw_text(draw, (28, summary_y + 94), f"initial error: {initial_proxy - initial_bs:+.6f}", "#333333", small_font)
    draw_text(
        draw,
        (450, summary_y + 28),
        "weighted closed-form errors: MAE/RMSE are value units; relative MAE uses denominator floor 0.01; relative plot is capped at 200%.",
        "#333333",
        small_font,
    )
    for row, (time, mae, rmse, rel_mae, clipped) in enumerate(metrics):
        draw_text(
            draw,
            (450, summary_y + 56 + row * 22),
            f"t={time:.2f}: MAE={mae:.5f}, RMSE={rmse:.5f}, rel_MAE={100.0 * rel_mae:.2f}%, clipped_bins={clipped}",
            "#333333",
            small_font,
        )

    image.save(path)
    return metrics


def main() -> None:
    params = GBMParams(n_paths=10_000_000, seed=7)
    start = perf_counter()
    times, coeffs_by_time, bin_edges_by_time, bin_sums_by_time, bin_counts_by_time = (
        fit_streaming_proxy(params)
    )
    metrics = write_plot(
        OUTPUT_PATH,
        times,
        coeffs_by_time,
        bin_edges_by_time,
        bin_sums_by_time,
        bin_counts_by_time,
        params,
    )

    initial_proxy = coeffs_by_time[0][0]
    initial_bs = float(black_scholes_value(np.array([params.s0]), params.maturity, params)[0])
    print()
    print("10M-path streaming proxy fit")
    print(
        f"default model: {proxy_model_description(params)}"
    )
    print(f"initial proxy:      {initial_proxy:.6f}")
    print(f"Black-Scholes:      {initial_bs:.6f}")
    print(f"initial error:      {initial_proxy - initial_bs:+.6f}")
    print()
    print("weighted closed-form Black-Scholes errors")
    print("relative error uses denominator max(Black-Scholes value, 0.01)")
    print("time    MAE       RMSE      rel MAE   clipped rel bins")
    print("----    ---       ----      -------   ----------------")
    for time, mae, rmse, rel_mae, clipped in metrics:
        print(f"{time:4.2f}    {mae:7.5f}   {rmse:7.5f}   {100.0 * rel_mae:7.2f}%   {clipped:16d}")
    print()
    print(f"plot written to: {OUTPUT_PATH.resolve()}")
    print(f"elapsed seconds: {perf_counter() - start:.1f}")


if __name__ == "__main__":
    main()
