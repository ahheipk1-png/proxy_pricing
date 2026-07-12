import csv
from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "tmp" / "european_vol_rate_proxy"
PLOT_PATH = OUTPUT_DIR / "european_vol_rate_proxy_slices.png"
CSV_PATH = OUTPUT_DIR / "european_vol_rate_proxy_metrics.csv"
SUMMARY_PATH = (
    ROOT
    / "Markdown"
    / "European"
    / "findings"
    / "european_vol_rate_proxy.md"
)


@dataclass(frozen=True)
class ExperimentConfig:
    strike: float = 100.0
    div_yield: float = 0.02
    maturity: float = 1.0
    d1_min: float = -3.5
    d1_max: float = 3.5
    vol_min: float = 0.08
    vol_max: float = 0.50
    rate_min: float = -0.01
    rate_max: float = 0.10
    train_d1_nodes: int = 19
    train_vol_nodes: int = 7
    train_rate_nodes: int = 7
    test_d1_nodes: int = 61
    test_vol_nodes: int = 13
    test_rate_nodes: int = 13
    mc_paths_per_state: int = 4096
    ridge: float = 1e-8
    rel_error_floor: float = 0.01
    seed: int = 20260712


@dataclass(frozen=True)
class ModelSpec:
    name: str
    coord: str
    target: str
    kind: str
    total_degree: int | None = None
    max_degrees: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class ScoreRow:
    option_type: str
    label_source: str
    model: str
    coordinate: str
    terms: int
    max_rel: float
    p99_rel: float
    avg_rel: float
    mae: float
    max_abs: float
    fit_seconds: float


@dataclass(frozen=True)
class LabelQualityRow:
    option_type: str
    max_rel: float
    p99_rel: float
    avg_rel: float
    mae: float
    max_abs: float


MODEL_SPECS = [
    ModelSpec("log sparse Chebyshev degree 5", "d1_vol_rate", "log", "sparse", 5),
    ModelSpec("log sparse Chebyshev degree 7", "d1_vol_rate", "log", "sparse", 7),
    ModelSpec("log sparse Chebyshev degree 9", "d1_vol_rate", "log", "sparse", 9),
    ModelSpec(
        "log anisotropic tensor Chebyshev 9x3x3",
        "d1_vol_rate",
        "log",
        "tensor",
        max_degrees=(9, 3, 3),
    ),
    ModelSpec("direct sparse Chebyshev degree 7", "d1_vol_rate", "direct", "sparse", 7),
    ModelSpec("log sparse Chebyshev degree 7", "logm_vol_rate", "log", "sparse", 7),
    ModelSpec("log sparse Chebyshev degree 7", "spot_vol_rate", "log", "sparse", 7),
]


def normal_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def normal_ppf(probability: np.ndarray) -> np.ndarray:
    """Acklam inverse-normal approximation, vectorized with NumPy only."""
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1.0 - 1e-15)
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )
    plow = 0.02425
    phigh = 1.0 - plow
    out = np.empty_like(p)

    low = p < plow
    high = p > phigh
    central = ~(low | high)

    q = np.sqrt(-2.0 * np.log(p[low]))
    out[low] = (
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )

    q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
    out[high] = -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )

    q = p[central] - 0.5
    r = q * q
    out[central] = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4])
            * r
            + 1.0
        )
    )
    return out


def one_dimensional_sobol_normals(n_paths: int, seed: int) -> np.ndarray:
    """Base-2 van der Corput sequence, which is the 1D Sobol sequence."""
    half = (n_paths + 1) // 2
    values = np.empty(half)
    for i in range(half):
        index = i + 1
        denominator = 2.0
        value = 0.0
        while index:
            index, bit = divmod(index, 2)
            value += bit / denominator
            denominator *= 2.0
        values[i] = value

    rng = np.random.default_rng(seed)
    values = (values + rng.random()) % 1.0
    base = normal_ppf(np.clip(values, 1e-12, 1.0 - 1e-12))
    normals = np.concatenate([base, -base])
    return normals[:n_paths]


def payoff(spot: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(spot - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - spot, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def spot_from_d1(
    d1: np.ndarray,
    vol: np.ndarray,
    rate: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    return cfg.strike * np.exp(
        d1 * vol * sqrt(cfg.maturity)
        - (rate - cfg.div_yield + 0.5 * vol**2) * cfg.maturity
    )


def d1_from_state(
    spot: np.ndarray,
    vol: np.ndarray,
    rate: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    return (
        np.log(spot / cfg.strike)
        + (rate - cfg.div_yield + 0.5 * vol**2) * cfg.maturity
    ) / (vol * sqrt(cfg.maturity))


def black_scholes_value(
    spot: np.ndarray,
    vol: np.ndarray,
    rate: np.ndarray,
    option_type: str,
    cfg: ExperimentConfig,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    sigma_sqrt_t = vol * sqrt(cfg.maturity)
    d1 = d1_from_state(spot, vol, rate, cfg)
    d2 = d1 - sigma_sqrt_t
    discounted_spot = spot * exp(-cfg.div_yield * cfg.maturity)
    discounted_strike = cfg.strike * np.exp(-rate * cfg.maturity)
    if option_type == "call":
        return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    if option_type == "put":
        return discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def build_state_grid(
    cfg: ExperimentConfig,
    d1_nodes: int,
    vol_nodes: int,
    rate_nodes: int,
    node_style: str = "uniform",
) -> dict[str, np.ndarray]:
    d1_axis = grid_axis(cfg.d1_min, cfg.d1_max, d1_nodes, node_style)
    vol_axis = grid_axis(cfg.vol_min, cfg.vol_max, vol_nodes, node_style)
    rate_axis = grid_axis(cfg.rate_min, cfg.rate_max, rate_nodes, node_style)
    d1, vol, rate = np.meshgrid(d1_axis, vol_axis, rate_axis, indexing="ij")
    spot = spot_from_d1(d1, vol, rate, cfg)
    return {
        "d1": d1.ravel(),
        "vol": vol.ravel(),
        "rate": rate.ravel(),
        "spot": spot.ravel(),
    }


def grid_axis(low: float, high: float, n_nodes: int, node_style: str) -> np.ndarray:
    if node_style == "uniform":
        return np.linspace(low, high, n_nodes)
    if node_style == "chebyshev":
        theta = np.linspace(np.pi, 0.0, n_nodes)
        unit_nodes = np.cos(theta)
        return 0.5 * (low + high) + 0.5 * (high - low) * unit_nodes
    raise ValueError(f"unknown node style: {node_style}")


def payoff_boundary_shift(
    spot: np.ndarray,
    vol: np.ndarray,
    rate: np.ndarray,
    option_type: str,
    cfg: ExperimentConfig,
) -> np.ndarray:
    threshold = (
        np.log(cfg.strike / spot)
        - (rate - cfg.div_yield - 0.5 * vol**2) * cfg.maturity
    ) / (vol * sqrt(cfg.maturity))
    if option_type == "call":
        return np.clip(threshold + 0.5, 0.0, 4.0)
    if option_type == "put":
        return np.clip(threshold - 0.5, -4.0, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def shifted_sobol_mc_value(
    state: dict[str, np.ndarray],
    option_type: str,
    cfg: ExperimentConfig,
) -> np.ndarray:
    values = np.empty_like(state["spot"])
    normals = one_dimensional_sobol_normals(cfg.mc_paths_per_state, cfg.seed)
    chunk_size = 160

    for start in range(0, values.size, chunk_size):
        end = min(start + chunk_size, values.size)
        spot = state["spot"][start:end]
        vol = state["vol"][start:end]
        rate = state["rate"][start:end]
        shift = payoff_boundary_shift(spot, vol, rate, option_type, cfg)
        shifted_normals = normals[None, :] + shift[:, None]
        likelihood_ratio = np.exp(
            -shift[:, None] * shifted_normals + 0.5 * shift[:, None] ** 2
        )
        growth = np.exp(
            (rate[:, None] - cfg.div_yield - 0.5 * vol[:, None] ** 2)
            * cfg.maturity
            + vol[:, None] * sqrt(cfg.maturity) * shifted_normals
        )
        terminal_spot = spot[:, None] * growth
        discounted_payoff = (
            np.exp(-rate[:, None] * cfg.maturity)
            * payoff(terminal_spot, cfg.strike, option_type)
            * likelihood_ratio
        )
        values[start:end] = discounted_payoff.mean(axis=1)
    return values


def scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(2.0 * (np.asarray(values, dtype=float) - low) / (high - low) - 1.0, -1.0, 1.0)


def feature_matrix(
    state: dict[str, np.ndarray],
    coord: str,
    cfg: ExperimentConfig,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    if coord == "d1_vol_rate":
        raw = np.column_stack([state["d1"], state["vol"], state["rate"]])
        default_bounds = {
            "x0": (cfg.d1_min, cfg.d1_max),
            "x1": (cfg.vol_min, cfg.vol_max),
            "x2": (cfg.rate_min, cfg.rate_max),
        }
    elif coord == "logm_vol_rate":
        raw = np.column_stack(
            [np.log(state["spot"] / cfg.strike), state["vol"], state["rate"]]
        )
        default_bounds = {
            f"x{i}": (float(raw[:, i].min()), float(raw[:, i].max()))
            for i in range(3)
        }
    elif coord == "spot_vol_rate":
        raw = np.column_stack([state["spot"] / cfg.strike - 1.0, state["vol"], state["rate"]])
        default_bounds = {
            f"x{i}": (float(raw[:, i].min()), float(raw[:, i].max()))
            for i in range(3)
        }
    else:
        raise ValueError(f"unknown coordinate system: {coord}")

    active_bounds = default_bounds if bounds is None else bounds
    scaled = np.column_stack(
        [
            scale(raw[:, i], active_bounds[f"x{i}"][0], active_bounds[f"x{i}"][1])
            for i in range(3)
        ]
    )
    return scaled, active_bounds


def sparse_terms(total_degree: int) -> list[tuple[int, int, int]]:
    return [
        (i, j, k)
        for i in range(total_degree + 1)
        for j in range(total_degree + 1 - i)
        for k in range(total_degree + 1 - i - j)
    ]


def tensor_terms(max_degrees: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    return [
        (i, j, k)
        for i in range(max_degrees[0] + 1)
        for j in range(max_degrees[1] + 1)
        for k in range(max_degrees[2] + 1)
    ]


def chebyshev_design(features: np.ndarray, terms: list[tuple[int, int, int]]) -> np.ndarray:
    max_degrees = [max(term[i] for term in terms) for i in range(3)]
    vanders = [chebvander(features[:, i], max_degrees[i]) for i in range(3)]
    design = np.empty((features.shape[0], len(terms)))
    for col, (i, j, k) in enumerate(terms):
        design[:, col] = vanders[0][:, i] * vanders[1][:, j] * vanders[2][:, k]
    return design


def fit_ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


class ChebyshevProxy:
    def __init__(
        self,
        spec: ModelSpec,
        cfg: ExperimentConfig,
        bounds: dict[str, tuple[float, float]],
        terms: list[tuple[int, int, int]],
        coeffs: np.ndarray,
    ) -> None:
        self.spec = spec
        self.cfg = cfg
        self.bounds = bounds
        self.terms = terms
        self.coeffs = coeffs

    def predict(self, state: dict[str, np.ndarray]) -> np.ndarray:
        features, _ = feature_matrix(state, self.spec.coord, self.cfg, self.bounds)
        raw = chebyshev_design(features, self.terms) @ self.coeffs
        if self.spec.target == "log":
            return np.maximum(np.exp(np.clip(raw, -30.0, 20.0)) - 1e-10, 0.0)
        return np.maximum(raw, 0.0)


def fit_proxy(
    state: dict[str, np.ndarray],
    values: np.ndarray,
    spec: ModelSpec,
    cfg: ExperimentConfig,
) -> tuple[ChebyshevProxy, float]:
    start = perf_counter()
    features, bounds = feature_matrix(state, spec.coord, cfg)
    if spec.kind == "sparse":
        terms = sparse_terms(int(spec.total_degree))
    elif spec.kind == "tensor":
        terms = tensor_terms(spec.max_degrees)
    else:
        raise ValueError(f"unknown model kind: {spec.kind}")

    design = chebyshev_design(features, terms)
    target = np.log(np.maximum(values, 0.0) + 1e-10) if spec.target == "log" else values
    coeffs = fit_ridge(design, target, cfg.ridge)
    return ChebyshevProxy(spec, cfg, bounds, terms, coeffs), perf_counter() - start


def score_prediction(
    prediction: np.ndarray,
    truth: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[float, float, float, float, float]:
    error = prediction - truth
    abs_error = np.abs(error)
    rel_error = abs_error / np.maximum(np.abs(truth), cfg.rel_error_floor)
    return (
        float(rel_error.max()),
        float(np.quantile(rel_error, 0.99)),
        float(rel_error.mean()),
        float(abs_error.mean()),
        float(abs_error.max()),
    )


def run_option(
    option_type: str, cfg: ExperimentConfig
) -> tuple[list[ScoreRow], LabelQualityRow, dict[str, object]]:
    train_state = build_state_grid(
        cfg,
        cfg.train_d1_nodes,
        cfg.train_vol_nodes,
        cfg.train_rate_nodes,
        node_style="chebyshev",
    )
    test_state = build_state_grid(
        cfg,
        cfg.test_d1_nodes,
        cfg.test_vol_nodes,
        cfg.test_rate_nodes,
        node_style="uniform",
    )
    truth_train = black_scholes_value(
        train_state["spot"], train_state["vol"], train_state["rate"], option_type, cfg
    )
    truth_test = black_scholes_value(
        test_state["spot"], test_state["vol"], test_state["rate"], option_type, cfg
    )
    mc_train = shifted_sobol_mc_value(train_state, option_type, cfg)
    label_max_rel, label_p99_rel, label_avg_rel, label_mae, label_max_abs = (
        score_prediction(mc_train, truth_train, cfg)
    )
    label_quality = LabelQualityRow(
        option_type=option_type,
        max_rel=label_max_rel,
        p99_rel=label_p99_rel,
        avg_rel=label_avg_rel,
        mae=label_mae,
        max_abs=label_max_abs,
    )

    rows: list[ScoreRow] = []
    fitted_models: dict[str, ChebyshevProxy] = {}
    for label_source, train_values in [
        ("exact labels", truth_train),
        ("shifted Sobol MC labels", mc_train),
    ]:
        for spec in MODEL_SPECS:
            model, fit_seconds = fit_proxy(train_state, train_values, spec, cfg)
            prediction = model.predict(test_state)
            max_rel, p99_rel, avg_rel, mae, max_abs = score_prediction(
                prediction, truth_test, cfg
            )
            rows.append(
                ScoreRow(
                    option_type=option_type,
                    label_source=label_source,
                    model=spec.name,
                    coordinate=spec.coord,
                    terms=len(model.terms),
                    max_rel=max_rel,
                    p99_rel=p99_rel,
                    avg_rel=avg_rel,
                    mae=mae,
                    max_abs=max_abs,
                    fit_seconds=fit_seconds,
                )
            )
            fitted_models[f"{label_source}|{spec.name}|{spec.coord}"] = model

    best_mc = min(
        (row for row in rows if row.label_source == "shifted Sobol MC labels"),
        key=lambda row: (row.p99_rel, row.max_rel),
    )
    model_key = f"{best_mc.label_source}|{best_mc.model}|{best_mc.coordinate}"
    return rows, label_quality, {
        "train_state": train_state,
        "test_state": test_state,
        "truth_test": truth_test,
        "best_mc": best_mc,
        "best_model": fitted_models[model_key],
    }


def write_csv(rows: list[ScoreRow]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "option_type",
                "label_source",
                "model",
                "coordinate",
                "terms",
                "max_rel",
                "p99_rel",
                "avg_rel",
                "mae",
                "max_abs",
                "fit_seconds",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.option_type,
                    row.label_source,
                    row.model,
                    row.coordinate,
                    row.terms,
                    f"{row.max_rel:.10f}",
                    f"{row.p99_rel:.10f}",
                    f"{row.avg_rel:.10f}",
                    f"{row.mae:.10f}",
                    f"{row.max_abs:.10f}",
                    f"{row.fit_seconds:.6f}",
                ]
            )


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, font) -> None:
    draw.text(xy, text, fill=fill, font=font)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    curves: list[tuple[np.ndarray, np.ndarray, str, str]],
    y_label: str,
    percent: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    draw.rectangle(box, fill="#ffffff", outline="#c7ccd4")
    draw_text(draw, (x0 + 10, y0 + 8), title, "#111111", font)

    left = x0 + 52
    right = x1 - 18
    top = y0 + 38
    bottom = y1 - 38
    all_x = np.concatenate([curve[0] for curve in curves])
    all_y = np.concatenate([curve[1] for curve in curves])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    if abs(y_max - y_min) < 1e-12:
        y_max += 1.0
        y_min -= 1.0
    margin = 0.08 * (y_max - y_min)
    y_min -= margin
    y_max += margin
    if y_min < 0.0 and y_max > 0.0:
        zero_y = bottom - (0.0 - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, zero_y, right, zero_y), fill="#e1e4e8")

    draw.line((left, bottom, right, bottom), fill="#333333")
    draw.line((left, top, left, bottom), fill="#333333")
    draw_text(draw, (left, bottom + 8), f"{x_min:.0f}", "#333333", small)
    draw_text(draw, (right - 28, bottom + 8), f"{x_max:.0f}", "#333333", small)
    y_top_text = f"{100.0 * y_max:.2f}%" if percent else f"{y_max:.3g}"
    y_bot_text = f"{100.0 * y_min:.2f}%" if percent else f"{y_min:.3g}"
    draw_text(draw, (x0 + 8, top - 6), y_top_text, "#333333", small)
    draw_text(draw, (x0 + 8, bottom - 10), y_bot_text, "#333333", small)
    draw_text(draw, (x0 + 8, y1 - 24), y_label, "#555555", small)

    for curve_index, (x_values, y_values, color, label) in enumerate(curves):
        points = []
        for xv, yv in zip(x_values, y_values):
            px = left + (xv - x_min) / (x_max - x_min) * (right - left)
            py = bottom - (yv - y_min) / (y_max - y_min) * (bottom - top)
            points.append((float(px), float(py)))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        lx = right - 150
        ly = top + 14 + 17 * curve_index
        draw.line((lx, ly + 5, lx + 24, ly + 5), fill=color, width=2)
        draw_text(draw, (lx + 30, ly), label, "#333333", small)


def slice_state(
    cfg: ExperimentConfig,
    d1_axis: np.ndarray,
    vol: float,
    rate: float,
) -> dict[str, np.ndarray]:
    vol_array = np.full_like(d1_axis, vol, dtype=float)
    rate_array = np.full_like(d1_axis, rate, dtype=float)
    return {
        "d1": d1_axis,
        "vol": vol_array,
        "rate": rate_array,
        "spot": spot_from_d1(d1_axis, vol_array, rate_array, cfg),
    }


def write_plot(cfg: ExperimentConfig, call_info: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model: ChebyshevProxy = call_info["best_model"]
    d1_axis = np.linspace(cfg.d1_min, cfg.d1_max, 241)
    width, height = 1760, 1140
    image = Image.new("RGB", (width, height), "#f5f6f8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    draw_text(draw, (30, 22), "European proxy with volatility and rate as features", "#111111", font)
    draw_text(
        draw,
        (30, 46),
        f"Best MC model for call: {model.spec.name}, coordinate={model.spec.coord}. Black lines are Black-Scholes, colored lines are proxy.",
        "#444444",
        small,
    )

    colors = ["#c0392b", "#1f77b4", "#2e7d32"]
    vol_slices = [0.10, 0.20, 0.40]
    rate_slices = [0.00, 0.05, 0.10]
    panel_w = 548
    panel_h = 330
    left = 30
    top = 88
    gap_x = 28
    gap_y = 34

    def make_curves_by_vol(metric: str) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
        curves = []
        for idx, vol in enumerate(vol_slices):
            state = slice_state(cfg, d1_axis, vol, 0.05)
            truth = black_scholes_value(state["spot"], state["vol"], state["rate"], "call", cfg)
            pred = model.predict(state)
            if metric == "value_truth":
                curves.append((state["spot"], truth, "#111111", f"BS vol={vol:.0%}"))
                curves.append((state["spot"], pred, colors[idx], f"proxy vol={vol:.0%}"))
            elif metric == "error":
                curves.append((state["spot"], pred - truth, colors[idx], f"vol={vol:.0%}"))
            elif metric == "rel":
                curves.append(
                    (
                        state["spot"],
                        np.abs(pred - truth) / np.maximum(np.abs(truth), cfg.rel_error_floor),
                        colors[idx],
                        f"vol={vol:.0%}",
                    )
                )
        return curves

    def make_curves_by_rate(metric: str) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
        curves = []
        for idx, rate in enumerate(rate_slices):
            state = slice_state(cfg, d1_axis, 0.20, rate)
            truth = black_scholes_value(state["spot"], state["vol"], state["rate"], "call", cfg)
            pred = model.predict(state)
            if metric == "value_truth":
                curves.append((state["spot"], truth, "#111111", f"BS r={rate:.0%}"))
                curves.append((state["spot"], pred, colors[idx], f"proxy r={rate:.0%}"))
            elif metric == "error":
                curves.append((state["spot"], pred - truth, colors[idx], f"r={rate:.0%}"))
            elif metric == "rel":
                curves.append(
                    (
                        state["spot"],
                        np.abs(pred - truth) / np.maximum(np.abs(truth), cfg.rel_error_floor),
                        colors[idx],
                        f"r={rate:.0%}",
                    )
                )
        return curves

    panels = [
        (left + 0 * (panel_w + gap_x), top, left + panel_w, top + panel_h),
        (left + 1 * (panel_w + gap_x), top, left + 1 * (panel_w + gap_x) + panel_w, top + panel_h),
        (left + 2 * (panel_w + gap_x), top, left + 2 * (panel_w + gap_x) + panel_w, top + panel_h),
        (left + 0 * (panel_w + gap_x), top + panel_h + gap_y, left + panel_w, top + 2 * panel_h + gap_y),
        (
            left + 1 * (panel_w + gap_x),
            top + panel_h + gap_y,
            left + 1 * (panel_w + gap_x) + panel_w,
            top + 2 * panel_h + gap_y,
        ),
        (
            left + 2 * (panel_w + gap_x),
            top + panel_h + gap_y,
            left + 2 * (panel_w + gap_x) + panel_w,
            top + 2 * panel_h + gap_y,
        ),
    ]

    draw_panel(draw, panels[0], "Value vs spot, varying vol at r=5%", make_curves_by_vol("value_truth"), "value")
    draw_panel(draw, panels[1], "Signed error, varying vol at r=5%", make_curves_by_vol("error"), "proxy - BS")
    draw_panel(draw, panels[2], "Relative error, varying vol at r=5%", make_curves_by_vol("rel"), "abs err / floor", True)
    draw_panel(draw, panels[3], "Value vs spot, varying rate at vol=20%", make_curves_by_rate("value_truth"), "value")
    draw_panel(draw, panels[4], "Signed error, varying rate at vol=20%", make_curves_by_rate("error"), "proxy - BS")
    draw_panel(draw, panels[5], "Relative error, varying rate at vol=20%", make_curves_by_rate("rel"), "abs err / floor", True)

    summary = [
        f"training grid: {cfg.train_d1_nodes} d1 x {cfg.train_vol_nodes} vol x {cfg.train_rate_nodes} rate = {cfg.train_d1_nodes * cfg.train_vol_nodes * cfg.train_rate_nodes} states",
        f"MC labels: {cfg.mc_paths_per_state:,} shifted 1D Sobol normals/state with likelihood-ratio correction",
        f"domain: d1 in [{cfg.d1_min:g}, {cfg.d1_max:g}], vol in [{cfg.vol_min:.0%}, {cfg.vol_max:.0%}], r in [{cfg.rate_min:.0%}, {cfg.rate_max:.0%}]",
        f"relative denominator floor: {cfg.rel_error_floor:g}",
    ]
    for i, line in enumerate(summary):
        draw_text(draw, (30, height - 118 + 22 * i), line, "#333333", small)

    image.save(PLOT_PATH)


def format_pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def write_summary(
    rows: list[ScoreRow],
    label_quality: list[LabelQualityRow],
    cfg: ExperimentConfig,
    elapsed_seconds: float,
) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: (row.option_type, row.label_source, row.p99_rel, row.max_rel))
    best_by_group = []
    for option_type in ["call", "put"]:
        for label_source in ["exact labels", "shifted Sobol MC labels"]:
            subset = [
                row
                for row in sorted_rows
                if row.option_type == option_type and row.label_source == label_source
            ]
            best_by_group.append(min(subset, key=lambda row: (row.p99_rel, row.max_rel)))

    lines = [
        "# European Vol/Rate Feature Proxy",
        "",
        "This experiment extends the European proxy from one feature to three features:",
        "",
        "- spot represented by the Black-Scholes `d1` coordinate",
        "- volatility",
        "- risk-free rate",
        "",
        "The goal is not to use Black-Scholes as the production method. Black-Scholes is",
        "used here as a clean benchmark so we can see whether the proxy surface learns",
        "the correct dependence on volatility and rate.",
        "",
        "## Setup",
        "",
        f"- strike: `{cfg.strike:g}`",
        f"- dividend yield: `{cfg.div_yield:.2%}`",
        f"- maturity: `{cfg.maturity:g}` year",
        f"- training domain: `d1 in [{cfg.d1_min:g}, {cfg.d1_max:g}]`, "
        f"`vol in [{cfg.vol_min:.2%}, {cfg.vol_max:.2%}]`, "
        f"`rate in [{cfg.rate_min:.2%}, {cfg.rate_max:.2%}]`",
        f"- training grid: `{cfg.train_d1_nodes} x {cfg.train_vol_nodes} x {cfg.train_rate_nodes}` "
        f"= `{cfg.train_d1_nodes * cfg.train_vol_nodes * cfg.train_rate_nodes}` Chebyshev-spaced states",
        f"- test grid: `{cfg.test_d1_nodes} x {cfg.test_vol_nodes} x {cfg.test_rate_nodes}` "
        f"= `{cfg.test_d1_nodes * cfg.test_vol_nodes * cfg.test_rate_nodes}` uniform states",
        f"- shifted Sobol MC paths per state: `{cfg.mc_paths_per_state:,}`",
        f"- relative error denominator: `max(true value, {cfg.rel_error_floor:g})`",
        f"- elapsed seconds: `{elapsed_seconds:.1f}`",
        "",
        "## MC Label Quality",
        "",
        "| Option | Max % Label Error | P99 % Label Error | Avg % Label Error | Label MAE | Label Max Abs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in label_quality:
        lines.append(
            "| "
            f"{row.option_type} | {format_pct(row.max_rel)} | {format_pct(row.p99_rel)} | "
            f"{format_pct(row.avg_rel)} | {row.mae:.6f} | {row.max_abs:.6f} |"
        )

    lines.extend(
        [
        "",
        "## Best Results By Group",
        "",
        "| Option | Label Source | Best Method | Coordinate | Terms | Max % Error | P99 % Error | Avg % Error | MAE | Fit Seconds |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_by_group:
        lines.append(
            "| "
            f"{row.option_type} | {row.label_source} | {row.model} | {row.coordinate} | "
            f"{row.terms} | {format_pct(row.max_rel)} | {format_pct(row.p99_rel)} | "
            f"{format_pct(row.avg_rel)} | {row.mae:.6f} | {row.fit_seconds:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Full Method Comparison",
            "",
            "| Option | Label Source | Method | Coordinate | Terms | Max % Error | P99 % Error | Avg % Error | MAE |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda row: (row.option_type, row.label_source, row.p99_rel, row.max_rel)):
        lines.append(
            "| "
            f"{row.option_type} | {row.label_source} | {row.model} | {row.coordinate} | "
            f"{row.terms} | {format_pct(row.max_rel)} | {format_pct(row.p99_rel)} | "
            f"{format_pct(row.avg_rel)} | {row.mae:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The most important feature-engineering result is that `d1` remains the right",
            "spot coordinate even when volatility and rate are state variables. It removes",
            "most of the strike/vol/time scaling from the spot axis, so the remaining",
            "surface in volatility and rate is smoother.",
            "",
            "The exact-label runs measure pure approximation error. The shifted-MC-label",
            "runs measure the realistic proxy workflow, where each training value is a",
            "Monte Carlo conditional expectation estimate. If MC errors dominate, the",
            "fix is usually more paths, better importance sampling, or fitting a residual",
            "around a simple analytic/control baseline.",
            "",
            "For this 3D European case, the generic candidate to carry forward is the",
            "log sparse Chebyshev proxy on `(d1, vol, rate)`. It is convex to train,",
            "fast to evaluate, and extends naturally to higher-dimensional feature sets",
            "without requiring a full tensor grid.",
            "",
            f"Diagnostic plot: `{PLOT_PATH}`",
            f"Raw metrics CSV: `{CSV_PATH}`",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines))


def main() -> None:
    cfg = ExperimentConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    all_rows: list[ScoreRow] = []
    label_quality: list[LabelQualityRow] = []
    plot_info = None

    for option_type in ["call", "put"]:
        rows, label_row, info = run_option(option_type, cfg)
        all_rows.extend(rows)
        label_quality.append(label_row)
        if option_type == "call":
            plot_info = info
        print(f"completed {option_type}: {len(rows)} model scores")

    elapsed_seconds = perf_counter() - start
    write_csv(all_rows)
    write_plot(cfg, plot_info)
    write_summary(all_rows, label_quality, cfg, elapsed_seconds)

    print()
    print("European vol/rate proxy experiment")
    print(
        f"training states: {cfg.train_d1_nodes * cfg.train_vol_nodes * cfg.train_rate_nodes:,}; "
        f"test states: {cfg.test_d1_nodes * cfg.test_vol_nodes * cfg.test_rate_nodes:,}"
    )
    print(f"shifted 1D Sobol MC paths/state: {cfg.mc_paths_per_state:,}")
    print()
    print("MC label quality at training states:")
    for row in label_quality:
        print(
            f"{row.option_type:4s}: max={format_pct(row.max_rel)}, "
            f"p99={format_pct(row.p99_rel)}, avg={format_pct(row.avg_rel)}, "
            f"MAE={row.mae:.6f}"
        )
    print()
    print("Best results by option and label source:")
    for option_type in ["call", "put"]:
        for label_source in ["exact labels", "shifted Sobol MC labels"]:
            subset = [
                row
                for row in all_rows
                if row.option_type == option_type and row.label_source == label_source
            ]
            best = min(subset, key=lambda row: (row.p99_rel, row.max_rel))
            print(
                f"{option_type:4s} | {label_source:24s} | "
                f"{best.model} ({best.coordinate}, {best.terms} terms): "
                f"max={format_pct(best.max_rel)}, p99={format_pct(best.p99_rel)}, "
                f"avg={format_pct(best.avg_rel)}, MAE={best.mae:.6f}"
            )

    print()
    print(f"CSV written to: {CSV_PATH}")
    print(f"plot written to: {PLOT_PATH}")
    print(f"summary written to: {SUMMARY_PATH}")
    print(f"elapsed seconds: {elapsed_seconds:.1f}")


if __name__ == "__main__":
    main()
