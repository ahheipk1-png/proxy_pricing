from math import exp, sqrt
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_stage1_smoothers import bspline_design, weighted_ridge
from plot_stage1_10m_proxy import RELATIVE_ERROR_FLOOR, draw_axes, draw_text
from plot_stage1_bspline_wide import draw_signed_error_panel, draw_wide_value_panel
from plot_stage1_relative_controlled_proxy import draw_relative_error_panel
from stage1_lsmc_european import (
    GBMParams,
    asymptotic_anchor_value,
    black_scholes_value,
    european_payoff,
    option_upper_bound,
)


OUTPUT_PATH = Path("stage1_mc_wing_shift_proxy_diagnostics.png")
PLOT_DELTA_LOW = 1e-4
PLOT_DELTA_HIGH = 1.0 - 1e-4
N_STATE_POINTS = 121
MC_PATHS_PER_STATE = 25_000
N_INTERNAL_KNOTS = 12
RIDGE = 1e-8
SHIFT_BUFFER = 0.5
SHIFT_CAP = 4.0


def d1_from_spot(spot: np.ndarray, tau: float, params: GBMParams) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return (spot - params.strike) / params.strike
    return (
        np.log(spot / params.strike)
        + (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    ) / (params.vol * sqrt(tau))


def spot_from_d1(d1: np.ndarray, tau: float, params: GBMParams) -> np.ndarray:
    d1 = np.asarray(d1, dtype=float)
    if tau <= 0.0:
        return params.strike * (1.0 + 0.01 * d1)
    return params.strike * np.exp(
        d1 * params.vol * sqrt(tau)
        - (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    )


def delta_space_spot_grid(
    tau: float,
    params: GBMParams,
    n_points: int,
    low_delta: float = PLOT_DELTA_LOW,
    high_delta: float = PLOT_DELTA_HIGH,
) -> np.ndarray:
    if tau <= 0.0:
        return np.linspace(50.0, 190.0, n_points)
    normal = NormalDist()
    d1_grid = np.linspace(normal.inv_cdf(low_delta), normal.inv_cdf(high_delta), n_points)
    return spot_from_d1(d1_grid, tau, params)


def payoff_boundary_shift(
    spot: np.ndarray, tau: float, params: GBMParams
) -> np.ndarray:
    """Normal shift for wing states, aimed toward the payoff boundary."""
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return np.zeros_like(spot)

    threshold = (
        np.log(params.strike / spot)
        - (params.rate - params.div_yield - 0.5 * params.vol**2) * tau
    ) / (params.vol * sqrt(tau))

    if params.option_type == "call":
        return np.clip(threshold + SHIFT_BUFFER, 0.0, SHIFT_CAP)
    if params.option_type == "put":
        return np.clip(threshold - SHIFT_BUFFER, -SHIFT_CAP, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def shifted_mc_option_value(
    spot: np.ndarray,
    tau: float,
    params: GBMParams,
    rng: np.random.Generator,
    n_paths: int = MC_PATHS_PER_STATE,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return european_payoff(spot, params)

    half_paths = (n_paths + 1) // 2
    base = rng.standard_normal((spot.size, half_paths))
    shift = payoff_boundary_shift(spot, tau, params)

    shifted_plus = shift[:, None] + base
    shifted_minus = shift[:, None] - base
    shifted_normals = np.concatenate((shifted_plus, shifted_minus), axis=1)[
        :, :n_paths
    ]

    likelihood_ratio = np.exp(
        -shift[:, None] * shifted_normals + 0.5 * shift[:, None] ** 2
    )
    growth = np.exp(
        (params.rate - params.div_yield - 0.5 * params.vol**2) * tau
        + params.vol * sqrt(tau) * shifted_normals
    )
    terminal_spot = spot[:, None] * growth
    payoff = european_payoff(terminal_spot, params)
    return exp(-params.rate * tau) * np.mean(payoff * likelihood_ratio, axis=1)


def fit_log_value_spline(
    spot: np.ndarray, values: np.ndarray, tau: float, params: GBMParams
) -> object:
    if tau <= 0.0:
        return lambda new_spot: european_payoff(np.asarray(new_spot, dtype=float), params)

    x = d1_from_spot(spot, tau, params)
    x_min = float(x.min())
    x_max = float(x.max())
    internal = np.linspace(x_min, x_max, N_INTERNAL_KNOTS + 2)[1:-1]
    knots = np.concatenate(([x_min] * 4, internal, [x_max] * 4))
    target = np.log(np.maximum(values, 0.0) + 1e-10)
    coeffs = weighted_ridge(
        bspline_design(x, knots), target, np.ones_like(values), RIDGE
    )

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_spot = np.asarray(new_spot, dtype=float)
        new_x = np.clip(d1_from_spot(new_spot, tau, params), x_min, x_max)
        raw = np.exp(bspline_design(new_x, knots) @ coeffs) - 1e-10
        lower = asymptotic_anchor_value(new_spot, tau, params)
        upper = option_upper_bound(new_spot, tau, params)
        return np.minimum(np.maximum(raw, lower), upper)

    return predict


def write_plot(path: Path, params: GBMParams) -> list[tuple[float, float, float, float]]:
    width = 2100
    height = 1980
    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    rng = np.random.default_rng(params.seed + 97)
    times = np.linspace(0.0, params.maturity, params.n_steps + 1)

    draw_text(draw, (28, 20), "MC wing-shifted proxy diagnostics", "#111111", font)
    draw_text(
        draw,
        (28, 42),
        "Training labels use shifted MC draws in the wings; closed form is diagnostics only.",
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
        state_spot = delta_space_spot_grid(tau, params, N_STATE_POINTS)
        state_value = shifted_mc_option_value(
            state_spot, tau, params, rng, MC_PATHS_PER_STATE
        )
        proxy = fit_log_value_spline(state_spot, state_value, tau, params)

        grid_spot = delta_space_spot_grid(tau, params, 501)
        proxy_values = proxy(grid_spot)
        closed_form_values = black_scholes_value(grid_spot, tau, params)
        signed_error = proxy_values - closed_form_values
        rel_error = np.abs(signed_error) / np.maximum(
            np.abs(closed_form_values), RELATIVE_ERROR_FLOOR
        )

        max_rel = float(rel_error.max())
        p99_rel = float(np.quantile(rel_error, 0.99))
        mae = float(np.mean(np.abs(signed_error)))
        max_abs = float(np.max(np.abs(signed_error)))
        metrics.append((time, max_rel, p99_rel, mae, max_abs))

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
            draw, panels[1], time, grid_spot, signed_error, font, small_font
        )
        draw_relative_error_panel(
            draw, panels[2], time, grid_spot, rel_error, font, small_font
        )

    summary_y = top + params.n_steps * panel_h + (params.n_steps - 1) * gap_y + 24
    draw_text(draw, (28, summary_y), "summary", "#111111", font)
    draw_text(
        draw,
        (28, summary_y + 28),
        f"state points per time: {N_STATE_POINTS}, MC paths per state: {MC_PATHS_PER_STATE:,}, internal knots: {N_INTERNAL_KNOTS}",
        "#333333",
        small_font,
    )
    draw_text(
        draw,
        (28, summary_y + 50),
        f"domain: delta {PLOT_DELTA_LOW:g} to {PLOT_DELTA_HIGH:g}; shift cap={SHIFT_CAP:g}; relative denominator floor={RELATIVE_ERROR_FLOOR:g}",
        "#333333",
        small_font,
    )
    draw_text(
        draw,
        (450, summary_y + 28),
        "The normal-shift likelihood ratio keeps the MC training labels unbiased.",
        "#333333",
        small_font,
    )
    for row, (time, max_rel, p99_rel, mae, max_abs) in enumerate(metrics):
        draw_text(
            draw,
            (450, summary_y + 56 + row * 22),
            f"t={time:.2f}: max_rel={100.0 * max_rel:.3f}%, p99_rel={100.0 * p99_rel:.3f}%, MAE={mae:.6f}, max_abs={max_abs:.6f}",
            "#333333",
            small_font,
        )

    image.save(path)
    return metrics


def main() -> None:
    params = GBMParams(n_paths=10_000_000, n_steps=5, seed=7, option_type="call")
    start = perf_counter()
    metrics = write_plot(OUTPUT_PATH, params)

    print("MC wing-shifted proxy diagnostics")
    print(
        f"method: {N_STATE_POINTS} delta-space states, "
        f"{MC_PATHS_PER_STATE:,} shifted MC paths/state, "
        f"log-value B-spline with {N_INTERNAL_KNOTS} internal knots"
    )
    print(f"relative denominator floor: {RELATIVE_ERROR_FLOOR:g}")
    print("time    max rel    p99 rel    MAE       max abs")
    print("----    -------    -------    ---       -------")
    for time, max_rel, p99_rel, mae, max_abs in metrics:
        print(
            f"{time:4.2f}    {100.0 * max_rel:7.3f}%   "
            f"{100.0 * p99_rel:7.3f}%   {mae:8.6f}  {max_abs:8.6f}"
        )
    print()
    print(f"plot written to: {OUTPUT_PATH.resolve()}")
    print(f"elapsed seconds: {perf_counter() - start:.1f}")


if __name__ == "__main__":
    main()
