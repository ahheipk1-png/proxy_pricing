from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_stage1_smoothers import binned_series, stream_training_targets
from experiment_stage1_accuracy import (
    piecewise_delta_poly_prediction,
    three_region_values,
)
from plot_stage1_10m_proxy import (
    RELATIVE_ERROR_CAP,
    RELATIVE_ERROR_FLOOR,
    draw_metric_panel,
    draw_text,
    draw_value_panel,
)
from stage1_lsmc_european import GBMParams, black_scholes_value


OUTPUT_PATH = Path("stage1_10m_piecewise_delta_diagnostics.png")
TAIL_DELTA = 0.001
TRAINED_DELTA = 0.01


def write_piecewise_plot(
    path: Path,
    times: np.ndarray,
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
        "10M-path piecewise-delta asymptotic proxy diagnostics",
        "#111111",
        font,
    )
    draw_text(
        draw,
        (28, 42),
        "Values show proxy, closed-form Black-Scholes, and binned training CE; errors compare proxy to closed form.",
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
        x_values, ce_values, counts = binned_series(
            bin_edges_by_time[step],
            bin_sums_by_time[step],
            bin_counts_by_time[step],
        )
        weights = counts.astype(float)
        closed_form_values = black_scholes_value(x_values, tau, params)

        learned_values = piecewise_delta_poly_prediction(
            x_values, ce_values, weights, tau, params
        )
        proxy_values = three_region_values(
            x_values,
            tau,
            learned_values,
            params,
            tail_delta=TAIL_DELTA,
            trained_delta=TRAINED_DELTA,
        )

        error = proxy_values - closed_form_values
        abs_error = np.abs(error)
        rel_error = abs_error / np.maximum(
            np.abs(closed_form_values), RELATIVE_ERROR_FLOOR
        )
        norm_weights = weights / weights.sum()
        mae = float(np.sum(norm_weights * abs_error))
        rmse = float(np.sqrt(np.sum(norm_weights * error**2)))
        rel_mae = float(np.sum(norm_weights * rel_error))

        draw_value_panel(
            draw,
            panels[0],
            time,
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
            time,
            "abs error",
            "|proxy-BS|",
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
            time,
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
        metrics.append((time, mae, rmse, rel_mae, clipped))

    summary_y = top + params.n_steps * panel_h + (params.n_steps - 1) * gap_y + 24
    draw_text(draw, (28, summary_y), "summary", "#111111", font)
    draw_text(draw, (28, summary_y + 28), f"training paths: {params.n_paths:,}", "#333333", small_font)
    draw_text(
        draw,
        (28, summary_y + 50),
        f"piecewise regions: delta [0.001,0.15], [0.15,0.85], [0.85,0.999]",
        "#333333",
        small_font,
    )
    draw_text(
        draw,
        (28, summary_y + 72),
        f"asymptotic mix: tail={TAIL_DELTA:g}, trained={TRAINED_DELTA:g}",
        "#333333",
        small_font,
    )
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
    params = GBMParams(n_paths=10_000_000, n_steps=5, seed=7, option_type="call")
    start = perf_counter()
    times, _path_coeffs, bin_edges_by_time, bin_sums_by_time, bin_counts_by_time = (
        stream_training_targets(params)
    )
    metrics = write_piecewise_plot(
        OUTPUT_PATH,
        times,
        bin_edges_by_time,
        bin_sums_by_time,
        bin_counts_by_time,
        params,
    )

    print()
    print("10M-path piecewise-delta asymptotic proxy diagnostics")
    print(f"tail delta: {TAIL_DELTA:g}; trained delta: {TRAINED_DELTA:g}")
    print("relative error uses denominator max(Black-Scholes value, 0.01)")
    print("time    MAE       RMSE      rel MAE   clipped rel bins")
    print("----    ---       ----      -------   ----------------")
    for time, mae, rmse, rel_mae, clipped in metrics:
        print(
            f"{time:4.2f}    {mae:7.5f}   {rmse:7.5f}   "
            f"{100.0 * rel_mae:7.2f}%   {clipped:16d}"
        )
    print()
    print(f"plot written to: {OUTPUT_PATH.resolve()}")
    print(f"elapsed seconds: {perf_counter() - start:.1f}")


if __name__ == "__main__":
    main()
