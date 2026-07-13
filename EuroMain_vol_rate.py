import csv
import os
from dataclasses import dataclass
from math import erf, exp, log, sqrt
from time import perf_counter

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Config:
    strike: float = 100.0
    div_yield: float = 0.02
    maturity: float = 1.0
    rel_error_floor: float = 0.01
    mc_paths_per_state: int = 2048
    ridge: float = 1e-8
    seed: int = 20260712


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_kind: str
    degree: int
    target_kind: str = "log"


@dataclass(frozen=True)
class ScoreRow:
    option_type: str
    label_source: str
    model: str
    feature_kind: str
    terms: int
    max_rel: float
    p99_rel: float
    avg_rel: float
    mae: float
    max_abs: float
    fit_seconds: float


@dataclass(frozen=True)
class ScenarioRow:
    option_type: str
    rate_curve: str
    vol_curve: str
    max_rel: float
    p99_rel: float
    avg_rel: float
    mae: float
    max_abs: float


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "tmp", "euro_vol_rate_term_structure")
METRICS_CSV = os.path.join(OUTPUT_DIR, "euro_vol_rate_term_structure_metrics.csv")
SCENARIO_CSV = os.path.join(OUTPUT_DIR, "euro_vol_rate_term_structure_scenarios.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "euro_vol_rate_term_structure_slices.png")
SUMMARY_PATH = os.path.join(
    ROOT,
    "Markdown",
    "European",
    "findings",
    "european_term_structure_proxy.md",
)

SEGMENT_X = np.array([-0.75, -0.25, 0.25, 0.75])
SEGMENT_X2 = SEGMENT_X * SEGMENT_X - np.mean(SEGMENT_X * SEGMENT_X)

MODEL_SPECS = [
    ModelSpec("collapsed effective curve, degree 5", "collapsed", 5),
    ModelSpec("collapsed effective curve, degree 7", "collapsed", 7),
    ModelSpec("curve stats, degree 3", "curve_stats", 3),
    ModelSpec("curve stats, degree 4", "curve_stats", 4),
    ModelSpec("full curve knots, degree 3", "full_curve", 3),
    ModelSpec("raw spot plus full curve, degree 3", "raw_spot_full_curve", 3),
]


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def normal_ppf(probability):
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
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )
    return out


def one_dimensional_sobol_normals(n_paths, seed):
    half = (int(n_paths) + 1) // 2
    uniforms = np.empty(half)
    for i in range(half):
        index = i + 1
        denominator = 2.0
        value = 0.0
        while index:
            index, bit = divmod(index, 2)
            value += bit / denominator
            denominator *= 2.0
        uniforms[i] = value
    rng = np.random.default_rng(seed)
    uniforms = (uniforms + rng.random()) % 1.0
    base = normal_ppf(uniforms)
    normals = np.concatenate([base, -base])
    return normals[: int(n_paths)]


def chebyshev_axis(low, high, n_nodes):
    theta = np.linspace(np.pi, 0.0, n_nodes)
    return 0.5 * (low + high) + 0.5 * (high - low) * np.cos(theta)


def payoff(spot, strike, option_type):
    if option_type == "call":
        return np.maximum(spot - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - spot, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def curve_stats(rates, vols):
    rates = np.asarray(rates, dtype=float)
    vols = np.asarray(vols, dtype=float)
    rate_avg = np.mean(rates, axis=1)
    rate_slope = rates[:, -1] - rates[:, 0]
    rate_curve = np.mean(rates[:, 1:3], axis=1) - 0.5 * (rates[:, 0] + rates[:, -1])
    vol_eff = np.sqrt(np.mean(vols * vols, axis=1))
    vol_slope = vols[:, -1] - vols[:, 0]
    vol_curve = np.mean(vols[:, 1:3], axis=1) - 0.5 * (vols[:, 0] + vols[:, -1])
    return {
        "rate_avg": rate_avg,
        "rate_slope": rate_slope,
        "rate_curve": rate_curve,
        "vol_eff": vol_eff,
        "vol_slope": vol_slope,
        "vol_curve": vol_curve,
    }


def integrated_terms(state, cfg):
    stats = curve_stats(state["rates"], state["vols"])
    rate_integral = stats["rate_avg"] * cfg.maturity
    variance_integral = stats["vol_eff"] ** 2 * cfg.maturity
    div_integral = cfg.div_yield * cfg.maturity
    return rate_integral, div_integral, variance_integral


def spot_from_d1(d1, rates, vols, cfg):
    state = {
        "d1": np.asarray(d1, dtype=float),
        "rates": np.asarray(rates, dtype=float),
        "vols": np.asarray(vols, dtype=float),
    }
    rate_integral, div_integral, variance_integral = integrated_terms(state, cfg)
    return cfg.strike * np.exp(
        state["d1"] * np.sqrt(variance_integral)
        - (rate_integral - div_integral + 0.5 * variance_integral)
    )


def black_scholes_term_structure_value(state, option_type, cfg):
    rate_integral, div_integral, variance_integral = integrated_terms(state, cfg)
    variance_sqrt = np.sqrt(np.maximum(variance_integral, 1e-16))
    d1 = state["d1"]
    d2 = d1 - variance_sqrt
    spot = state["spot"]
    discounted_spot = spot * np.exp(-div_integral)
    discounted_strike = cfg.strike * np.exp(-rate_integral)
    if option_type == "call":
        return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    if option_type == "put":
        return discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def shifted_mc_term_structure_value(state, option_type, cfg):
    rate_integral, div_integral, variance_integral = integrated_terms(state, cfg)
    variance_sqrt = np.sqrt(np.maximum(variance_integral, 1e-16))
    spot = state["spot"]
    threshold = (
        np.log(cfg.strike / spot)
        - rate_integral
        + div_integral
        + 0.5 * variance_integral
    ) / variance_sqrt
    if option_type == "call":
        shift = np.clip(threshold + 0.5, 0.0, 4.0)
    elif option_type == "put":
        shift = np.clip(threshold - 0.5, -4.0, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    normals = one_dimensional_sobol_normals(cfg.mc_paths_per_state, cfg.seed)
    values = np.empty_like(spot)
    chunk_size = 192
    for start in range(0, spot.size, chunk_size):
        end = min(start + chunk_size, spot.size)
        shifted_normals = normals[None, :] + shift[start:end, None]
        likelihood_ratio = np.exp(
            -shift[start:end, None] * shifted_normals + 0.5 * shift[start:end, None] ** 2
        )
        terminal = spot[start:end, None] * np.exp(
            rate_integral[start:end, None]
            - div_integral
            - 0.5 * variance_integral[start:end, None]
            + variance_sqrt[start:end, None] * shifted_normals
        )
        samples = (
            np.exp(-rate_integral[start:end, None])
            * payoff(terminal, cfg.strike, option_type)
            * likelihood_ratio
        )
        values[start:end] = np.mean(samples, axis=1)
    return values


def make_state(d1_values, rates, vols, cfg, rate_name=None, vol_name=None):
    d1_values = np.asarray(d1_values, dtype=float)
    rates = np.asarray(rates, dtype=float)
    vols = np.asarray(vols, dtype=float)
    if rates.ndim == 1:
        rates = np.repeat(rates[None, :], d1_values.size, axis=0)
    if vols.ndim == 1:
        vols = np.repeat(vols[None, :], d1_values.size, axis=0)
    spot = spot_from_d1(d1_values, rates, vols, cfg)
    state = {"d1": d1_values, "rates": rates, "vols": vols, "spot": spot}
    if rate_name is not None:
        state["rate_name"] = np.array([rate_name] * d1_values.size)
    if vol_name is not None:
        state["vol_name"] = np.array([vol_name] * d1_values.size)
    return state


def concatenate_states(states):
    result = {}
    for key in ["d1", "rates", "vols", "spot", "rate_name", "vol_name"]:
        if key in states[0]:
            result[key] = np.concatenate([state[key] for state in states], axis=0)
    return result


def training_state(cfg):
    d1_axis = chebyshev_axis(-3.5, 3.5, 17)
    rate_level_axis = chebyshev_axis(-0.005, 0.09, 5)
    rate_slope_axis = chebyshev_axis(-0.045, 0.045, 3)
    rate_curve_axis = chebyshev_axis(-0.035, 0.035, 3)
    vol_level_axis = chebyshev_axis(0.10, 0.50, 5)
    vol_slope_axis = chebyshev_axis(-0.45, 0.45, 3)
    vol_curve_axis = chebyshev_axis(-0.30, 0.30, 3)

    states = []
    for rate_level in rate_level_axis:
        for rate_slope in rate_slope_axis:
            for rate_curve in rate_curve_axis:
                rates = rate_level + rate_slope * SEGMENT_X + rate_curve * SEGMENT_X2
                for vol_level in vol_level_axis:
                    for vol_slope in vol_slope_axis:
                        for vol_curve in vol_curve_axis:
                            vols = vol_level * np.exp(
                                vol_slope * SEGMENT_X + vol_curve * SEGMENT_X2
                            )
                            states.append(make_state(d1_axis, rates, vols, cfg))
    return concatenate_states(states)


def named_rate_curves():
    return {
        "flat_low": np.array([0.00, 0.00, 0.00, 0.00]),
        "flat_mid": np.array([0.04, 0.04, 0.04, 0.04]),
        "flat_high": np.array([0.09, 0.09, 0.09, 0.09]),
        "upward": np.array([-0.005, 0.025, 0.055, 0.085]),
        "downward": np.array([0.085, 0.055, 0.025, -0.005]),
        "humped": np.array([0.015, 0.075, 0.075, 0.015]),
        "inverted": np.array([0.10, 0.06, 0.02, 0.00]),
    }


def named_vol_curves():
    return {
        "flat_low": np.array([0.10, 0.10, 0.10, 0.10]),
        "flat_mid": np.array([0.22, 0.22, 0.22, 0.22]),
        "flat_high": np.array([0.45, 0.45, 0.45, 0.45]),
        "upward": np.array([0.11, 0.18, 0.30, 0.45]),
        "downward": np.array([0.45, 0.30, 0.18, 0.11]),
        "humped": np.array([0.14, 0.36, 0.36, 0.14]),
        "u_shape": np.array([0.38, 0.18, 0.18, 0.38]),
    }


def test_state(cfg):
    d1_axis = np.linspace(-3.5, 3.5, 81)
    states = []
    for rate_name, rates in named_rate_curves().items():
        for vol_name, vols in named_vol_curves().items():
            states.append(make_state(d1_axis, rates, vols, cfg, rate_name, vol_name))
    return concatenate_states(states)


def feature_matrix(state, feature_kind):
    stats = curve_stats(state["rates"], state["vols"])
    if feature_kind == "collapsed":
        return np.column_stack([state["d1"], stats["rate_avg"], stats["vol_eff"]])
    if feature_kind == "curve_stats":
        return np.column_stack(
            [
                state["d1"],
                stats["rate_avg"],
                stats["rate_slope"],
                stats["rate_curve"],
                stats["vol_eff"],
                stats["vol_slope"],
                stats["vol_curve"],
            ]
        )
    if feature_kind == "full_curve":
        return np.column_stack([state["d1"], state["rates"], state["vols"]])
    if feature_kind == "raw_spot_full_curve":
        return np.column_stack([state["spot"] / 100.0 - 1.0, state["rates"], state["vols"]])
    raise ValueError(f"unknown feature kind: {feature_kind}")


def sparse_terms(dimension, degree):
    terms = []

    def rec(prefix, remaining_dim, remaining_degree):
        if remaining_dim == 1:
            for power in range(remaining_degree + 1):
                terms.append(tuple(prefix + [power]))
            return
        for power in range(remaining_degree + 1):
            rec(prefix + [power], remaining_dim - 1, remaining_degree - power)

    rec([], dimension, degree)
    return terms


def scale_features(features, bounds=None):
    if bounds is None:
        lows = np.min(features, axis=0)
        highs = np.max(features, axis=0)
        span = np.maximum(highs - lows, 1e-12)
        bounds = (lows, highs, span)
    lows, highs, span = bounds
    scaled = 2.0 * (features - lows) / span - 1.0
    return np.clip(scaled, -1.0, 1.0), bounds


def chebyshev_design(features, terms):
    max_degrees = [max(term[col] for term in terms) for col in range(features.shape[1])]
    vanders = [chebvander(features[:, col], max_degrees[col]) for col in range(features.shape[1])]
    design = np.empty((features.shape[0], len(terms)))
    for col_index, term in enumerate(terms):
        column = np.ones(features.shape[0])
        for feature_index, power in enumerate(term):
            column *= vanders[feature_index][:, power]
        design[:, col_index] = column
    return design


def fit_ridge(design, target, ridge):
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


class SparseChebyshevProxy:
    def __init__(self, spec, terms, bounds, coeffs):
        self.spec = spec
        self.terms = terms
        self.bounds = bounds
        self.coeffs = coeffs

    def predict(self, state):
        features = feature_matrix(state, self.spec.feature_kind)
        scaled, _ = scale_features(features, self.bounds)
        raw = chebyshev_design(scaled, self.terms) @ self.coeffs
        if self.spec.target_kind == "log":
            return np.maximum(np.exp(np.clip(raw, -30.0, 20.0)) - 1e-10, 0.0)
        return np.maximum(raw, 0.0)


def fit_proxy(train_state, train_values, spec, cfg):
    start = perf_counter()
    features = feature_matrix(train_state, spec.feature_kind)
    scaled, bounds = scale_features(features)
    terms = sparse_terms(scaled.shape[1], spec.degree)
    design = chebyshev_design(scaled, terms)
    target = np.log(np.maximum(train_values, 0.0) + 1e-10)
    coeffs = fit_ridge(design, target, cfg.ridge)
    return SparseChebyshevProxy(spec, terms, bounds, coeffs), perf_counter() - start


def score(prediction, truth, cfg):
    error = prediction - truth
    absolute = np.abs(error)
    relative = absolute / np.maximum(np.abs(truth), cfg.rel_error_floor)
    return (
        float(np.max(relative)),
        float(np.quantile(relative, 0.99)),
        float(np.mean(relative)),
        float(np.mean(absolute)),
        float(np.max(absolute)),
    )


def scenario_scores(state, prediction, truth, option_type, cfg):
    rows = []
    for rate_name in sorted(set(state["rate_name"])):
        for vol_name in sorted(set(state["vol_name"])):
            mask = (state["rate_name"] == rate_name) & (state["vol_name"] == vol_name)
            max_rel, p99_rel, avg_rel, mae, max_abs = score(prediction[mask], truth[mask], cfg)
            rows.append(
                ScenarioRow(
                    option_type=option_type,
                    rate_curve=str(rate_name),
                    vol_curve=str(vol_name),
                    max_rel=max_rel,
                    p99_rel=p99_rel,
                    avg_rel=avg_rel,
                    mae=mae,
                    max_abs=max_abs,
                )
            )
    return rows


def run_option(option_type, cfg):
    train = training_state(cfg)
    test = test_state(cfg)
    exact_train = black_scholes_term_structure_value(train, option_type, cfg)
    mc_train = shifted_mc_term_structure_value(train, option_type, cfg)
    exact_test = black_scholes_term_structure_value(test, option_type, cfg)
    label_quality = score(mc_train, exact_train, cfg)

    scores = []
    models = {}
    for label_source, train_values in [
        ("exact labels", exact_train),
        ("shifted Sobol MC labels", mc_train),
    ]:
        for spec in MODEL_SPECS:
            proxy, fit_seconds = fit_proxy(train, train_values, spec, cfg)
            prediction = proxy.predict(test)
            max_rel, p99_rel, avg_rel, mae, max_abs = score(prediction, exact_test, cfg)
            scores.append(
                ScoreRow(
                    option_type=option_type,
                    label_source=label_source,
                    model=spec.name,
                    feature_kind=spec.feature_kind,
                    terms=len(proxy.terms),
                    max_rel=max_rel,
                    p99_rel=p99_rel,
                    avg_rel=avg_rel,
                    mae=mae,
                    max_abs=max_abs,
                    fit_seconds=fit_seconds,
                )
            )
            models[(label_source, spec.name, spec.feature_kind)] = proxy

    best_mc = min(
        [row for row in scores if row.label_source == "shifted Sobol MC labels"],
        key=lambda row: (row.p99_rel, row.max_rel),
    )
    best_model = models[(best_mc.label_source, best_mc.model, best_mc.feature_kind)]
    best_prediction = best_model.predict(test)
    scenario_rows = scenario_scores(test, best_prediction, exact_test, option_type, cfg)
    return scores, label_quality, scenario_rows, best_model


def shape_invariance_checks(cfg):
    d1_axis = np.linspace(-2.5, 2.5, 41)
    rate_level = 0.04
    vol_eff = 0.25
    rate_up = rate_level + 0.04 * SEGMENT_X
    rate_down = rate_level - 0.04 * SEGMENT_X
    variance_up = vol_eff**2 + 0.025 * SEGMENT_X
    variance_down = vol_eff**2 - 0.025 * SEGMENT_X
    vol_up = np.sqrt(np.maximum(variance_up, 1e-8))
    vol_down = np.sqrt(np.maximum(variance_down, 1e-8))
    up_state = make_state(d1_axis, rate_up, vol_up, cfg)
    down_state = make_state(d1_axis, rate_down, vol_down, cfg)
    rows = {}
    for option_type in ["call", "put"]:
        up = black_scholes_term_structure_value(up_state, option_type, cfg)
        down = black_scholes_term_structure_value(down_state, option_type, cfg)
        rows[option_type] = float(np.max(np.abs(up - down)))
    return rows


def write_metrics_csv(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METRICS_CSV, "w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "option_type",
                "label_source",
                "model",
                "feature_kind",
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
                    row.feature_kind,
                    row.terms,
                    f"{row.max_rel:.10f}",
                    f"{row.p99_rel:.10f}",
                    f"{row.avg_rel:.10f}",
                    f"{row.mae:.10f}",
                    f"{row.max_abs:.10f}",
                    f"{row.fit_seconds:.6f}",
                ]
            )


def write_scenario_csv(rows):
    with open(SCENARIO_CSV, "w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "option_type",
                "rate_curve",
                "vol_curve",
                "max_rel",
                "p99_rel",
                "avg_rel",
                "mae",
                "max_abs",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.option_type,
                    row.rate_curve,
                    row.vol_curve,
                    f"{row.max_rel:.10f}",
                    f"{row.p99_rel:.10f}",
                    f"{row.avg_rel:.10f}",
                    f"{row.mae:.10f}",
                    f"{row.max_abs:.10f}",
                ]
            )


def pct(value):
    return f"{100.0 * value:.3f}%"


def draw_text(draw, xy, text, fill, font):
    draw.text(xy, text, fill=fill, font=font)


def draw_panel(draw, box, title, series, y_label, percent=False):
    x0, y0, x1, y1 = box
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    draw.rectangle(box, fill="#ffffff", outline="#c7ccd4")
    draw_text(draw, (x0 + 10, y0 + 8), title, "#111111", font)
    left, right = x0 + 54, x1 - 18
    top, bottom = y0 + 38, y1 - 38
    all_x = np.concatenate([item[0] for item in series])
    all_y = np.concatenate([item[1] for item in series])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0
    margin = 0.08 * (y_max - y_min)
    y_min -= margin
    y_max += margin
    if y_min < 0.0 < y_max:
        zero_y = bottom - (0.0 - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, zero_y, right, zero_y), fill="#e0e4ea")
    draw.line((left, bottom, right, bottom), fill="#222222")
    draw.line((left, top, left, bottom), fill="#222222")
    draw_text(draw, (left, bottom + 8), f"{x_min:.0f}", "#444444", small)
    draw_text(draw, (right - 34, bottom + 8), f"{x_max:.0f}", "#444444", small)
    y_top = f"{100.0 * y_max:.2f}%" if percent else f"{y_max:.3g}"
    y_bottom = f"{100.0 * y_min:.2f}%" if percent else f"{y_min:.3g}"
    draw_text(draw, (x0 + 8, top - 6), y_top, "#444444", small)
    draw_text(draw, (x0 + 8, bottom - 10), y_bottom, "#444444", small)
    draw_text(draw, (x0 + 8, y1 - 24), y_label, "#555555", small)

    for index, (x_values, y_values, color, label) in enumerate(series):
        points = []
        for xv, yv in zip(x_values, y_values):
            px = left + (xv - x_min) / (x_max - x_min) * (right - left)
            py = bottom - (yv - y_min) / (y_max - y_min) * (bottom - top)
            points.append((float(px), float(py)))
        draw.line(points, fill=color, width=2)
        lx = right - 158
        ly = top + 12 + 17 * index
        draw.line((lx, ly + 5, lx + 24, ly + 5), fill=color, width=2)
        draw_text(draw, (lx + 30, ly), label, "#333333", small)


def write_plot(best_model, cfg):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    d1_axis = np.linspace(-3.2, 3.2, 181)
    scenarios = [
        ("flat/flat", named_rate_curves()["flat_mid"], named_vol_curves()["flat_mid"], "#c0392b"),
        ("up/up", named_rate_curves()["upward"], named_vol_curves()["upward"], "#1f77b4"),
        ("down/down", named_rate_curves()["downward"], named_vol_curves()["downward"], "#2e7d32"),
        ("hump/hump", named_rate_curves()["humped"], named_vol_curves()["humped"], "#8e44ad"),
    ]
    width, height = 1760, 720
    image = Image.new("RGB", (width, height), "#f5f6f8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    draw_text(draw, (30, 22), "European term-structure proxy", "#111111", font)
    draw_text(
        draw,
        (30, 46),
        "Black lines are generalized Black-Scholes; colored lines are the proxy trained on shifted Sobol MC labels.",
        "#444444",
        small,
    )

    value_series = []
    error_series = []
    rel_series = []
    for label, rates, vols, color in scenarios:
        state = make_state(d1_axis, rates, vols, cfg)
        truth = black_scholes_term_structure_value(state, "call", cfg)
        pred = best_model.predict(state)
        value_series.append((state["spot"], truth, "#111111", f"BS {label}"))
        value_series.append((state["spot"], pred, color, f"proxy {label}"))
        error_series.append((state["spot"], pred - truth, color, label))
        rel_series.append(
            (
                state["spot"],
                np.abs(pred - truth) / np.maximum(np.abs(truth), cfg.rel_error_floor),
                color,
                label,
            )
        )

    panel_w = 548
    panel_h = 430
    left = 30
    top = 88
    gap = 28
    panels = [
        (left, top, left + panel_w, top + panel_h),
        (left + panel_w + gap, top, left + 2 * panel_w + gap, top + panel_h),
        (left + 2 * (panel_w + gap), top, left + 3 * panel_w + 2 * gap, top + panel_h),
    ]
    draw_panel(draw, panels[0], "Call value vs spot", value_series, "value")
    draw_panel(draw, panels[1], "Signed error", error_series, "proxy - BS")
    draw_panel(draw, panels[2], "Relative error", rel_series, "abs err / floor", True)

    footer = [
        "European with deterministic rate and volatility term structures.",
        "For this payoff, only integrated rate and integrated variance are theoretically needed.",
        f"Training labels use {cfg.mc_paths_per_state:,} shifted Sobol terminal normals per state.",
    ]
    for index, line in enumerate(footer):
        draw_text(draw, (30, height - 92 + 22 * index), line, "#333333", small)
    image.save(PLOT_PATH)


def write_summary(rows, label_quality, scenario_rows, invariance, cfg, elapsed):
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    best_rows = []
    for option_type in ["call", "put"]:
        for label_source in ["exact labels", "shifted Sobol MC labels"]:
            subset = [
                row
                for row in rows
                if row.option_type == option_type and row.label_source == label_source
            ]
            best_rows.append(min(subset, key=lambda row: (row.p99_rel, row.max_rel)))

    lines = [
        "# European Term-Structure Vol/Rate Proxy",
        "",
        "This standalone experiment is run by `EuroMain_vol_rate.py`. It keeps",
        "`EuroMain.py` unchanged.",
        "",
        "The experiment asks whether a European proxy should use the whole rate and",
        "volatility term structure as features, or whether lower-dimensional effective",
        "features are enough.",
        "",
        "For deterministic rates and deterministic volatility, the European terminal",
        "log-price is normal with",
        "",
        "```text",
        "R = integral r(t) dt,    Q = integral q(t) dt,    V = integral sigma(t)^2 dt.",
        "```",
        "",
        "Therefore the option value depends on the curve shapes only through `R` and",
        "`V`. The tested collapsed feature vector is:",
        "",
        "```text",
        "(d1, average rate, effective volatility)",
        "```",
        "",
        "where `effective volatility = sqrt(V / T)`.",
        "",
        "## Setup",
        "",
        "- four piecewise-constant rate buckets over one year",
        "- four piecewise-constant volatility buckets over one year",
        "- training states: `17 d1 nodes x 5 rate levels x 3 rate slopes x 3 rate curvatures x 5 vol levels x 3 vol slopes x 3 vol curvatures = 6,885`",
        "- test states: `81 d1 nodes x 7 named rate curves x 7 named vol curves = 3,969` per option type",
        "- MC labels: shifted one-dimensional Sobol terminal-normal draws with likelihood-ratio correction",
        "- benchmark: generalized Black-Scholes with deterministic term structures",
        f"- shifted Sobol paths per state: `{cfg.mc_paths_per_state:,}`",
        f"- elapsed seconds: `{elapsed:.1f}`",
        "",
        "## MC Label Quality",
        "",
        "| Option | Max % Label Error | P99 % Label Error | Avg % Label Error | MAE | Max Abs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for option_type, values in label_quality.items():
        lines.append(
            f"| {option_type} | {pct(values[0])} | {pct(values[1])} | "
            f"{pct(values[2])} | {values[3]:.6f} | {values[4]:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best Results",
            "",
            "| Option | Label Source | Best Method | Feature Kind | Terms | Max % Error | P99 % Error | Avg % Error | MAE |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_rows:
        lines.append(
            f"| {row.option_type} | {row.label_source} | {row.model} | {row.feature_kind} | "
            f"{row.terms} | {pct(row.max_rel)} | {pct(row.p99_rel)} | "
            f"{pct(row.avg_rel)} | {row.mae:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Full Method Comparison",
            "",
            "| Option | Label Source | Method | Feature Kind | Terms | Max % Error | P99 % Error | Avg % Error | MAE |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda row: (row.option_type, row.label_source, row.p99_rel, row.max_rel)):
        lines.append(
            f"| {row.option_type} | {row.label_source} | {row.model} | {row.feature_kind} | "
            f"{row.terms} | {pct(row.max_rel)} | {pct(row.p99_rel)} | "
            f"{pct(row.avg_rel)} | {row.mae:.6f} |"
        )

    worst_scenarios = sorted(scenario_rows, key=lambda row: row.max_rel, reverse=True)[:12]
    lines.extend(
        [
            "",
            "## Worst Named Term-Structure Cases",
            "",
            "These rows use the best shifted-MC model for each option type.",
            "",
            "| Option | Rate Curve | Vol Curve | Max % Error | P99 % Error | Avg % Error | MAE |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in worst_scenarios:
        lines.append(
            f"| {row.option_type} | {row.rate_curve} | {row.vol_curve} | "
            f"{pct(row.max_rel)} | {pct(row.p99_rel)} | {pct(row.avg_rel)} | {row.mae:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Shape Invariance Check",
            "",
            "The check below compares upward and downward term structures with the same",
            "average rate and the same integrated variance. For a European option, the",
            "values should be identical up to floating-point noise.",
            "",
            "| Option | Max Absolute Difference |",
            "|---|---:|",
        ]
    )
    for option_type, value in invariance.items():
        lines.append(f"| {option_type} | {value:.12f} |")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "For European options under deterministic rate and volatility curves, do not",
            "feed the whole term structure into the proxy by default. The generic and",
            "more stable feature set is `(d1, average rate, effective volatility)`, where",
            "`d1` is computed from integrated drift and integrated variance.",
            "",
            "Raw term-structure knots are not wrong, but they add redundant dimensions.",
            "That makes sparse regression work harder and is a bad habit to carry into",
            "higher-dimensional exotics unless the payoff really observes the path at",
            "intermediate dates.",
            "",
            f"Diagnostic plot: `{PLOT_PATH}`",
            f"Metrics CSV: `{METRICS_CSV}`",
            f"Scenario CSV: `{SCENARIO_CSV}`",
            "",
        ]
    )
    with open(SUMMARY_PATH, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines))


def main():
    cfg = Config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = perf_counter()
    all_rows = []
    all_scenarios = []
    label_quality = {}
    best_call_model = None

    for option_type in ["call", "put"]:
        rows, labels, scenario_rows, best_model = run_option(option_type, cfg)
        all_rows.extend(rows)
        all_scenarios.extend(scenario_rows)
        label_quality[option_type] = labels
        if option_type == "call":
            best_call_model = best_model
        print(f"completed {option_type}: {len(rows)} model scores")

    invariance = shape_invariance_checks(cfg)
    elapsed = perf_counter() - start
    write_metrics_csv(all_rows)
    write_scenario_csv(all_scenarios)
    write_plot(best_call_model, cfg)
    write_summary(all_rows, label_quality, all_scenarios, invariance, cfg, elapsed)

    print()
    print("European term-structure vol/rate proxy")
    print("old EuroMain.py was not modified")
    print("training states: 6,885")
    print("test states per option: 3,969")
    print(f"shifted Sobol MC paths/state: {cfg.mc_paths_per_state:,}")
    print()
    print("MC label quality:")
    for option_type, values in label_quality.items():
        print(
            f"{option_type:4s}: max={pct(values[0])}, p99={pct(values[1])}, "
            f"avg={pct(values[2])}, MAE={values[3]:.6f}"
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
                f"{best.model} ({best.feature_kind}, {best.terms} terms): "
                f"max={pct(best.max_rel)}, p99={pct(best.p99_rel)}, "
                f"avg={pct(best.avg_rel)}, MAE={best.mae:.6f}"
            )

    print()
    print("Shape invariance check, same average rate and integrated variance:")
    for option_type, value in invariance.items():
        print(f"{option_type:4s}: max absolute difference={value:.12f}")
    print()
    print(f"summary written to: {SUMMARY_PATH}")
    print(f"plot written to: {PLOT_PATH}")
    print(f"metrics written to: {METRICS_CSV}")
    print(f"elapsed seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
