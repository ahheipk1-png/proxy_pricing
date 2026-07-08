import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from PIL import Image, ImageDraw, ImageFont

from compare_stage1_smoothers import (
    ModelFit,
    bspline_design,
    fit_pchip,
    loess_predict,
    natural_cubic_design,
    weighted_ridge,
)
from plot_stage1_10m_proxy import RELATIVE_ERROR_FLOOR, draw_text
from plot_stage1_bspline_wide import draw_signed_error_panel, draw_wide_value_panel
from plot_stage1_mc_wing_shift_proxy import (
    MC_PATHS_PER_STATE,
    N_STATE_POINTS,
    d1_from_spot,
    delta_space_spot_grid,
    shifted_mc_option_value,
)
from plot_stage1_relative_controlled_proxy import draw_relative_error_panel
from stage1_lsmc_european import GBMParams, black_scholes_value, european_payoff


OUTPUT_PATH = Path("stage1_mc_tail_biased_no_asym_diagnostics.png")
CSV_PATH = Path("stage1_mc_tail_biased_no_asym_bakeoff.csv")
PLOT_METHOD = ("log Chebyshev", "d1, degree=7")


@dataclass(frozen=True)
class ScoreRow:
    method: str
    detail: str
    time: float
    max_rel: float
    p99_rel: float
    mae: float
    max_abs: float


def model_key(model: ModelFit) -> tuple[str, str]:
    return model.name, model.detail


def x_coord(spot: np.ndarray, tau: float, params: GBMParams, coord: str) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    if coord == "d1":
        return d1_from_spot(spot, tau, params)
    if coord == "spot":
        return spot / params.s0 - 1.0
    if coord == "logm":
        return np.log(spot / params.strike)
    raise ValueError(f"unknown coordinate: {coord}")


def nonnegative(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def scaled_unit_transform(x: np.ndarray):
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    def transform(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.clip(2.0 * (values - x_min) / (x_max - x_min) - 1.0, -1.0, 1.0)

    return transform


def fit_polynomial(
    spot: np.ndarray,
    target: np.ndarray,
    tau: float,
    params: GBMParams,
    coord: str,
    degree: int,
    target_kind: str,
) -> ModelFit:
    x = x_coord(spot, tau, params, coord)
    design = np.column_stack([x**power for power in range(degree + 1)])
    y = (
        np.log(np.maximum(target, 0.0) + 1e-10)
        if target_kind == "log"
        else target
    )
    coeffs = weighted_ridge(design, y, np.ones_like(target), 1e-8)

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_x = x_coord(new_spot, tau, params, coord)
        new_design = np.column_stack([new_x**power for power in range(degree + 1)])
        raw = new_design @ coeffs
        if target_kind == "log":
            return np.exp(raw) - 1e-10
        return nonnegative(raw)

    return ModelFit(
        name=f"{target_kind} polynomial",
        detail=f"{coord}, degree={degree}",
        predict=predict,
    )


def fit_chebyshev_model(
    spot: np.ndarray,
    target: np.ndarray,
    tau: float,
    params: GBMParams,
    coord: str,
    degree: int,
    target_kind: str,
) -> ModelFit:
    x = x_coord(spot, tau, params, coord)
    transform = scaled_unit_transform(x)
    u = transform(x)
    y = (
        np.log(np.maximum(target, 0.0) + 1e-10)
        if target_kind == "log"
        else target
    )
    coeffs = weighted_ridge(chebvander(u, degree), y, np.ones_like(target), 1e-8)

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_u = transform(x_coord(new_spot, tau, params, coord))
        raw = chebvander(new_u, degree) @ coeffs
        if target_kind == "log":
            return np.exp(raw) - 1e-10
        return nonnegative(raw)

    return ModelFit(
        name=f"{target_kind} Chebyshev",
        detail=f"{coord}, degree={degree}",
        predict=predict,
    )


def fourier_design(u: np.ndarray, order: int, full: bool) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    columns = [np.ones_like(u)]
    for k in range(1, order + 1):
        columns.append(np.cos(k * np.pi * u))
        if full:
            columns.append(np.sin(k * np.pi * u))
    return np.column_stack(columns)


def fit_fourier_model(
    spot: np.ndarray,
    target: np.ndarray,
    tau: float,
    params: GBMParams,
    coord: str,
    order: int,
    target_kind: str,
    full: bool,
) -> ModelFit:
    x = x_coord(spot, tau, params, coord)
    transform = scaled_unit_transform(x)
    u = transform(x)
    y = (
        np.log(np.maximum(target, 0.0) + 1e-10)
        if target_kind == "log"
        else target
    )
    coeffs = weighted_ridge(
        fourier_design(u, order, full), y, np.ones_like(target), 1e-8
    )

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_u = transform(x_coord(new_spot, tau, params, coord))
        raw = fourier_design(new_u, order, full) @ coeffs
        if target_kind == "log":
            return np.exp(raw) - 1e-10
        return nonnegative(raw)

    basis_name = "full Fourier" if full else "cosine Fourier"
    return ModelFit(
        name=f"{target_kind} {basis_name}",
        detail=f"{coord}, order={order}",
        predict=predict,
    )


def fit_bspline_model(
    spot: np.ndarray,
    target: np.ndarray,
    tau: float,
    params: GBMParams,
    coord: str,
    n_knots: int,
    target_kind: str,
) -> ModelFit:
    x = x_coord(spot, tau, params, coord)
    x_min = float(x.min())
    x_max = float(x.max())
    internal = np.linspace(x_min, x_max, n_knots + 2)[1:-1]
    knots = np.concatenate(([x_min] * 4, internal, [x_max] * 4))
    y = (
        np.log(np.maximum(target, 0.0) + 1e-10)
        if target_kind == "log"
        else target
    )
    coeffs = weighted_ridge(bspline_design(x, knots), y, np.ones_like(target), 1e-8)

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_x = np.clip(x_coord(new_spot, tau, params, coord), x_min, x_max)
        raw = bspline_design(new_x, knots) @ coeffs
        if target_kind == "log":
            return np.exp(raw) - 1e-10
        return nonnegative(raw)

    return ModelFit(
        name=f"{target_kind} B-spline",
        detail=f"{coord}, knots={n_knots}",
        predict=predict,
    )


def fit_natural_cubic_model(
    spot: np.ndarray,
    target: np.ndarray,
    tau: float,
    params: GBMParams,
    coord: str,
    n_knots: int,
    target_kind: str,
) -> ModelFit:
    x = x_coord(spot, tau, params, coord)
    x_min = float(x.min())
    x_max = float(x.max())
    knots = np.linspace(x_min, x_max, n_knots + 2)
    y = (
        np.log(np.maximum(target, 0.0) + 1e-10)
        if target_kind == "log"
        else target
    )
    coeffs = weighted_ridge(
        natural_cubic_design(x, knots),
        y,
        np.ones_like(target),
        1e-8,
        unpenalized_columns=2,
    )

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_x = np.clip(x_coord(new_spot, tau, params, coord), x_min, x_max)
        raw = natural_cubic_design(new_x, knots) @ coeffs
        if target_kind == "log":
            return np.exp(raw) - 1e-10
        return nonnegative(raw)

    return ModelFit(
        name=f"{target_kind} natural cubic",
        detail=f"{coord}, knots={n_knots}",
        predict=predict,
    )


def fit_pchip_model(
    spot: np.ndarray,
    target: np.ndarray,
    tau: float,
    params: GBMParams,
    coord: str,
    target_kind: str,
) -> ModelFit:
    x = x_coord(spot, tau, params, coord)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = target[order]
    y = (
        np.log(np.maximum(y_sorted, 0.0) + 1e-10)
        if target_kind == "log"
        else y_sorted
    )
    model = fit_pchip(x_sorted, y)
    x_min = float(x_sorted.min())
    x_max = float(x_sorted.max())

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_x = np.clip(x_coord(new_spot, tau, params, coord), x_min, x_max)
        raw = model.predict(new_x)
        if target_kind == "log":
            return np.exp(raw) - 1e-10
        return nonnegative(raw)

    return ModelFit(
        name=f"{target_kind} PCHIP",
        detail=coord,
        predict=predict,
    )


def fit_loess_model(
    spot: np.ndarray,
    target: np.ndarray,
    tau: float,
    params: GBMParams,
    coord: str,
    frac: float,
    target_kind: str,
) -> ModelFit:
    x = x_coord(spot, tau, params, coord)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = target[order]
    y = (
        np.log(np.maximum(y_sorted, 0.0) + 1e-10)
        if target_kind == "log"
        else y_sorted
    )
    x_min = float(x_sorted.min())
    x_max = float(x_sorted.max())

    def predict(new_spot: np.ndarray) -> np.ndarray:
        new_x = np.clip(x_coord(new_spot, tau, params, coord), x_min, x_max)
        raw = loess_predict(
            x_sorted,
            y,
            np.ones_like(y),
            new_x,
            frac,
            degree=2,
        )
        if target_kind == "log":
            return np.exp(raw) - 1e-10
        return nonnegative(raw)

    return ModelFit(
        name=f"{target_kind} LOESS",
        detail=f"{coord}, frac={frac:g}",
        predict=predict,
    )


def build_models(
    spot: np.ndarray, target: np.ndarray, tau: float, params: GBMParams
) -> list[ModelFit]:
    models = []
    for coord in ["d1", "logm", "spot"]:
        for degree in [5, 7, 9, 11]:
            models.append(fit_polynomial(spot, target, tau, params, coord, degree, "direct"))
            models.append(fit_polynomial(spot, target, tau, params, coord, degree, "log"))
        for degree in [5, 7, 9, 11, 13, 15, 18, 22]:
            models.append(fit_chebyshev_model(spot, target, tau, params, coord, degree, "direct"))
            models.append(fit_chebyshev_model(spot, target, tau, params, coord, degree, "log"))
        for order in [3, 5, 7, 9, 11, 14, 18]:
            models.append(fit_fourier_model(spot, target, tau, params, coord, order, "log", False))
            models.append(fit_fourier_model(spot, target, tau, params, coord, order, "log", True))
        for n_knots in [6, 8, 10, 12, 16, 20, 28]:
            models.append(fit_bspline_model(spot, target, tau, params, coord, n_knots, "direct"))
            models.append(fit_bspline_model(spot, target, tau, params, coord, n_knots, "log"))
        for n_knots in [6, 10, 14, 18]:
            models.append(fit_natural_cubic_model(spot, target, tau, params, coord, n_knots, "direct"))
            models.append(fit_natural_cubic_model(spot, target, tau, params, coord, n_knots, "log"))
        models.append(fit_pchip_model(spot, target, tau, params, coord, "direct"))
        models.append(fit_pchip_model(spot, target, tau, params, coord, "log"))
        for frac in [0.14, 0.22, 0.35]:
            models.append(fit_loess_model(spot, target, tau, params, coord, frac, "direct"))
            models.append(fit_loess_model(spot, target, tau, params, coord, frac, "log"))
    return models


def score_model(
    model: ModelFit, tau: float, params: GBMParams, time: float
) -> ScoreRow:
    grid_spot = delta_space_spot_grid(tau, params, 501)
    truth = black_scholes_value(grid_spot, tau, params)
    prediction = np.asarray(model.predict(grid_spot), dtype=float)
    error = prediction - truth
    rel_error = np.abs(error) / np.maximum(np.abs(truth), RELATIVE_ERROR_FLOOR)
    return ScoreRow(
        method=model.name,
        detail=model.detail,
        time=time,
        max_rel=float(rel_error.max()),
        p99_rel=float(np.quantile(rel_error, 0.99)),
        mae=float(np.mean(np.abs(error))),
        max_abs=float(np.max(np.abs(error))),
    )


def aggregate_scores(scores: list[ScoreRow]) -> list[dict[str, object]]:
    aggregates = []
    for key in sorted({(score.method, score.detail) for score in scores}):
        rows = [score for score in scores if (score.method, score.detail) == key]
        aggregates.append(
            {
                "method": key[0],
                "detail": key[1],
                "worst_max_rel": max(row.max_rel for row in rows),
                "avg_p99_rel": float(np.mean([row.p99_rel for row in rows])),
                "avg_mae": float(np.mean([row.mae for row in rows])),
                "worst_max_abs": max(row.max_abs for row in rows),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            row["worst_max_rel"],
            row["avg_p99_rel"],
            row["avg_mae"],
        ),
    )


def write_csv(scores: list[ScoreRow], aggregates: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "detail",
                "time",
                "max_rel",
                "p99_rel",
                "mae",
                "max_abs",
            ]
        )
        for score in scores:
            writer.writerow(
                [
                    score.method,
                    score.detail,
                    f"{score.time:.6f}",
                    f"{score.max_rel:.10f}",
                    f"{score.p99_rel:.10f}",
                    f"{score.mae:.10f}",
                    f"{score.max_abs:.10f}",
                ]
            )
        writer.writerow([])
        writer.writerow(
            [
                "aggregate_method",
                "detail",
                "worst_max_rel",
                "avg_p99_rel",
                "avg_mae",
                "worst_max_abs",
            ]
        )
        for row in aggregates:
            writer.writerow(
                [
                    row["method"],
                    row["detail"],
                    f"{row['worst_max_rel']:.10f}",
                    f"{row['avg_p99_rel']:.10f}",
                    f"{row['avg_mae']:.10f}",
                    f"{row['worst_max_abs']:.10f}",
                ]
            )


def select_model(
    models: list[ModelFit], desired_key: tuple[str, str]
) -> ModelFit:
    for model in models:
        if model_key(model) == desired_key:
            return model
    raise ValueError(f"missing plot method: {desired_key}")


def write_plot(
    path: Path,
    params: GBMParams,
    labels_by_time: dict[float, tuple[np.ndarray, np.ndarray]],
) -> list[ScoreRow]:
    width = 2100
    height = 1980
    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    times = np.linspace(0.0, params.maturity, params.n_steps + 1)

    draw_text(draw, (28, 20), "MC tail-biased proxy without asymptotic proxy", "#111111", font)
    draw_text(
        draw,
        (28, 42),
        f"Plot method: {PLOT_METHOD[0]} ({PLOT_METHOD[1]}). Training labels use shifted MC only.",
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
        if tau <= 0.0:
            model = ModelFit(
                name="payoff",
                detail="maturity",
                predict=lambda new_spot: european_payoff(
                    np.asarray(new_spot, dtype=float), params
                ),
            )
        else:
            state_spot, state_value = labels_by_time[time]
            model = select_model(build_models(state_spot, state_value, tau, params), PLOT_METHOD)

        grid_spot = delta_space_spot_grid(tau, params, 501)
        proxy_values = np.asarray(model.predict(grid_spot), dtype=float)
        closed_form_values = black_scholes_value(grid_spot, tau, params)
        signed_error = proxy_values - closed_form_values
        rel_error = np.abs(signed_error) / np.maximum(
            np.abs(closed_form_values), RELATIVE_ERROR_FLOOR
        )
        metrics.append(
            ScoreRow(
                method=PLOT_METHOD[0],
                detail=PLOT_METHOD[1],
                time=time,
                max_rel=float(rel_error.max()),
                p99_rel=float(np.quantile(rel_error, 0.99)),
                mae=float(np.mean(np.abs(signed_error))),
                max_abs=float(np.max(np.abs(signed_error))),
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
        f"state points per time: {N_STATE_POINTS}, MC paths per state: {MC_PATHS_PER_STATE:,}",
        "#333333",
        small_font,
    )
    draw_text(
        draw,
        (28, summary_y + 50),
        f"no asymptotic proxy, no asymptotic anchors, no asymptotic clipping; relative denominator floor={RELATIVE_ERROR_FLOOR:g}",
        "#333333",
        small_font,
    )
    for row, score in enumerate(metrics):
        draw_text(
            draw,
            (450, summary_y + 28 + row * 22),
            f"t={score.time:.2f}: max_rel={100.0 * score.max_rel:.3f}%, p99_rel={100.0 * score.p99_rel:.3f}%, MAE={score.mae:.6f}, max_abs={score.max_abs:.6f}",
            "#333333",
            small_font,
        )

    image.save(path)
    return metrics


def main() -> None:
    params = GBMParams(n_paths=10_000_000, n_steps=5, seed=7, option_type="call")
    rng = np.random.default_rng(params.seed + 197)
    times = np.linspace(0.0, params.maturity, params.n_steps + 1)
    start = perf_counter()
    labels_by_time = {}
    scores = []

    for step in range(1, params.n_steps + 1):
        time = float(times[step])
        tau = float(params.maturity - time)
        if tau <= 0.0:
            continue

        state_spot = delta_space_spot_grid(tau, params, N_STATE_POINTS)
        state_value = shifted_mc_option_value(
            state_spot, tau, params, rng, MC_PATHS_PER_STATE
        )
        labels_by_time[time] = (state_spot, state_value)
        models = build_models(state_spot, state_value, tau, params)
        for model in models:
            scores.append(score_model(model, tau, params, time))

    aggregates = aggregate_scores(scores)
    write_csv(scores, aggregates)
    plot_scores = write_plot(OUTPUT_PATH, params, labels_by_time)

    print("MC tail-biased no-asym proxy bakeoff")
    print(
        f"training: {N_STATE_POINTS} delta-space states/time, "
        f"{MC_PATHS_PER_STATE:,} shifted MC paths/state"
    )
    print("no asymptotic proxy, anchors, mixing, or asymptotic clipping")
    print()
    print("Top aggregate methods, sorted by worst-time max relative error:")
    print("worst max    avg p99     avg MAE    method")
    for row in aggregates[:12]:
        print(
            f"{100.0 * row['worst_max_rel']:8.3f}%  "
            f"{100.0 * row['avg_p99_rel']:8.3f}%  "
            f"{row['avg_mae']:9.6f}  {row['method']} | {row['detail']}"
        )

    print()
    print(f"plotted method: {PLOT_METHOD[0]} | {PLOT_METHOD[1]}")
    print("time    max rel    p99 rel    MAE       max abs")
    print("----    -------    -------    ---       -------")
    for score in plot_scores:
        print(
            f"{score.time:4.2f}    {100.0 * score.max_rel:7.3f}%   "
            f"{100.0 * score.p99_rel:7.3f}%   {score.mae:8.6f}  {score.max_abs:8.6f}"
        )
    print()
    print(f"CSV written to: {CSV_PATH.resolve()}")
    print(f"plot written to: {OUTPUT_PATH.resolve()}")
    print(f"elapsed seconds: {perf_counter() - start:.1f}")


if __name__ == "__main__":
    main()
