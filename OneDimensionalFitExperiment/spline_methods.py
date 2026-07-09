"""Dependency-free one-dimensional smoothers used by the expanded study."""

from math import comb

import numpy as np
from numpy.polynomial.chebyshev import chebvander


EPS = 1e-12


def _ordered_unique(x, y):
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    x, index = np.unique(x, return_index=True)
    return x, y[index]


def _scale_x(x):
    low, high = float(np.min(x)), float(np.max(x))
    width = max(high - low, EPS)
    return (np.asarray(x) - low) / width, low, width


def _solve(design, y, penalty):
    normal = design.T @ design + penalty
    return np.linalg.solve(normal + 1e-12 * np.eye(len(normal)), design.T @ y)


def _gcv(design, y, penalty):
    normal = design.T @ design + penalty + 1e-12 * np.eye(design.shape[1])
    inverse = np.linalg.inv(normal)
    coefficients = inverse @ design.T @ y
    residual = y - design @ coefficients
    degrees = float(np.trace(inverse @ (design.T @ design)))
    denominator = max(1.0 - degrees / len(y), 1e-6)
    return float(np.mean(residual * residual) / denominator**2), coefficients


def _endpoint_slope(slope, secant):
    if slope * secant <= 0.0:
        return 0.0
    return np.sign(slope) * min(abs(slope), 3.0 * abs(secant))


def pchip_slopes(x, y):
    h = np.diff(x)
    delta = np.diff(y) / h
    if len(x) == 2:
        return np.array([delta[0], delta[0]])
    slopes = np.zeros_like(y)
    same_sign = delta[:-1] * delta[1:] > 0.0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    denominator = (
        w1 / np.where(delta[:-1] == 0.0, 1.0, delta[:-1])
        + w2 / np.where(delta[1:] == 0.0, 1.0, delta[1:])
    )
    interior = np.zeros_like(denominator)
    np.divide(w1 + w2, denominator, out=interior, where=np.abs(denominator) > EPS)
    slopes[1:-1] = np.where(same_sign, interior, 0.0)
    slopes[0] = _endpoint_slope(
        ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]),
        delta[0],
    )
    slopes[-1] = _endpoint_slope(
        ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2])
        / (h[-1] + h[-2]),
        delta[-1],
    )
    return slopes


def akima_slopes(x, y):
    if len(x) < 5:
        return pchip_slopes(x, y)
    delta = np.diff(y) / np.diff(x)
    extended = np.empty(len(delta) + 4)
    extended[2:-2] = delta
    extended[1] = 2.0 * delta[0] - delta[1]
    extended[0] = 2.0 * extended[1] - delta[0]
    extended[-2] = 2.0 * delta[-1] - delta[-2]
    extended[-1] = 2.0 * extended[-2] - delta[-1]
    slopes = np.empty_like(y)
    for index in range(len(y)):
        left_far, left = extended[index : index + 2]
        right, right_far = extended[index + 2 : index + 4]
        w_left = abs(right_far - right)
        w_right = abs(left - left_far)
        slopes[index] = (
            (w_left * left + w_right * right) / (w_left + w_right)
            if w_left + w_right > EPS
            else 0.5 * (left + right)
        )
    return slopes


def makima_slopes(x, y):
    if len(x) < 5:
        return pchip_slopes(x, y)
    delta = np.diff(y) / np.diff(x)
    extended = np.empty(len(delta) + 4)
    extended[2:-2] = delta
    extended[1] = 2.0 * delta[0] - delta[1]
    extended[0] = 2.0 * extended[1] - delta[0]
    extended[-2] = 2.0 * delta[-1] - delta[-2]
    extended[-1] = 2.0 * extended[-2] - delta[-1]
    slopes = np.empty_like(y)
    for index in range(len(y)):
        left_far, left = extended[index : index + 2]
        right, right_far = extended[index + 2 : index + 4]
        w_left = abs(right_far - right) + 0.5 * abs(right_far + right)
        w_right = abs(left - left_far) + 0.5 * abs(left + left_far)
        slopes[index] = (
            (w_left * left + w_right * right) / (w_left + w_right)
            if w_left + w_right > EPS
            else 0.5 * (left + right)
        )
    return slopes


def hermite_predictor(x, y, slopes):
    def predict(new_x):
        new_x = np.asarray(new_x, dtype=float)
        index = np.clip(np.searchsorted(x, new_x) - 1, 0, len(x) - 2)
        h = x[index + 1] - x[index]
        t = np.clip((new_x - x[index]) / h, 0.0, 1.0)
        return (
            (2.0 * t**3 - 3.0 * t**2 + 1.0) * y[index]
            + (t**3 - 2.0 * t**2 + t) * h * slopes[index]
            + (-2.0 * t**3 + 3.0 * t**2) * y[index + 1]
            + (t**3 - t**2) * h * slopes[index + 1]
        )

    return predict


def bspline_design(unit_x, n_basis, degree=3, interior_knots=None):
    unit_x = np.clip(np.asarray(unit_x, dtype=float), 0.0, 1.0)
    if interior_knots is None:
        count = n_basis - degree - 1
        interior_knots = np.linspace(0.0, 1.0, count + 2)[1:-1]
    interior_knots = np.asarray(interior_knots, dtype=float)
    knots = np.r_[np.zeros(degree + 1), interior_knots, np.ones(degree + 1)]
    basis = np.zeros((len(unit_x), len(knots) - 1))
    for index in range(len(knots) - 1):
        basis[:, index] = (unit_x >= knots[index]) & (unit_x < knots[index + 1])
    basis[unit_x == 1.0, len(knots) - degree - 2] = 1.0
    for level in range(1, degree + 1):
        next_basis = np.zeros((len(unit_x), len(knots) - level - 1))
        for index in range(next_basis.shape[1]):
            left_width = knots[index + level] - knots[index]
            right_width = knots[index + level + 1] - knots[index + 1]
            if left_width > 0.0:
                next_basis[:, index] += (
                    (unit_x - knots[index]) / left_width * basis[:, index]
                )
            if right_width > 0.0:
                next_basis[:, index] += (
                    (knots[index + level + 1] - unit_x)
                    / right_width
                    * basis[:, index + 1]
                )
        basis = next_basis
    return basis


def _linear_smoother(x, y, family):
    unit, low, width = _scale_x(x)
    n = len(x)
    if family == "bspline_regression":
        candidates = [min(value, n - 1) for value in (8, 12, 20, 32)]
        candidates = sorted(set(value for value in candidates if value >= 4))
        fits = []
        for n_basis in candidates:
            design = bspline_design(unit, n_basis)
            penalty = 1e-8 * np.eye(design.shape[1])
            score, coefficients = _gcv(design, y, penalty)
            fits.append((score, n_basis, coefficients))
        _, n_basis, coefficients = min(fits, key=lambda item: item[0])

        def predict(new_x):
            return bspline_design((np.asarray(new_x) - low) / width, n_basis) @ coefficients

        return predict

    n_basis = min(30, max(8, n // 3))
    design = bspline_design(unit, n_basis)
    difference = np.diff(np.eye(n_basis), n=2, axis=0)
    lambdas = np.logspace(-7, 5, 17)
    fits = []
    for smoothing in lambdas:
        penalty = smoothing * (difference.T @ difference)
        score, coefficients = _gcv(design, y, penalty)
        fits.append((score, smoothing, coefficients))
    _, smoothing, coefficients = min(fits, key=lambda item: item[0])

    if family == "adaptive_pspline":
        for _ in range(4):
            curvature = difference @ coefficients
            local_weight = 1.0 / np.sqrt(curvature * curvature + 1e-4)
            local_weight /= np.mean(local_weight)
            penalty = smoothing * (
                difference.T @ (local_weight[:, None] * difference)
            )
            coefficients = _solve(design, y, penalty)

    def predict(new_x):
        return bspline_design((np.asarray(new_x) - low) / width, n_basis) @ coefficients

    return predict


def _natural_kernel(a, b):
    return np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :]) ** 3


def _natural_spline(x, y, smoothing):
    unit, low, width = _scale_x(x)
    kernel = _natural_kernel(unit, unit)
    linear = np.column_stack((np.ones(len(unit)), unit))
    system = np.block(
        [
            [kernel + smoothing * np.eye(len(unit)), linear],
            [linear.T, np.zeros((2, 2))],
        ]
    )
    rhs = np.r_[y, np.zeros(2)]
    coefficients = np.linalg.solve(system + 1e-12 * np.eye(len(system)), rhs)
    radial, affine = coefficients[: len(unit)], coefficients[len(unit) :]

    def predict(new_x):
        new_unit = np.clip((np.asarray(new_x) - low) / width, 0.0, 1.0)
        return _natural_kernel(new_unit, unit) @ radial + affine[0] + affine[1] * new_unit

    return predict


def _cv_natural_smoothing(x, y):
    candidates = np.r_[0.0, np.logspace(-9, 0, 10)]
    fold = np.arange(len(x)) % 5
    scores = []
    for smoothing in candidates:
        errors = []
        for held_out in range(5):
            train = fold != held_out
            curve = _natural_spline(x[train], y[train], smoothing)
            errors.extend((curve(x[~train]) - y[~train]) ** 2)
        scores.append(float(np.mean(errors)))
    return _natural_spline(x, y, float(candidates[int(np.argmin(scores))]))


def _natural_regression(x, y):
    unit, low, width = _scale_x(x)
    candidates = [5, 9, 15, 25]
    fits = []
    for count in candidates:
        knots = np.linspace(0.0, 1.0, min(count, len(x) - 2) + 2)[1:-1]
        design = np.column_stack((np.ones(len(x)), unit, _natural_kernel(unit, knots)))
        penalty = np.diag(np.r_[0.0, 0.0, np.full(len(knots), 1e-7)])
        score, coefficients = _gcv(design, y, penalty)
        fits.append((score, knots, coefficients))
    _, knots, coefficients = min(fits, key=lambda item: item[0])

    def predict(new_x):
        new_unit = np.clip((np.asarray(new_x) - low) / width, 0.0, 1.0)
        design = np.column_stack(
            (np.ones(len(new_unit)), new_unit, _natural_kernel(new_unit, knots))
        )
        return design @ coefficients

    return predict


def _adaptive_knot_regression(x, y):
    unit, low, width = _scale_x(x)
    pilot = _linear_smoother(x, y, "pspline")
    pilot_y = pilot(x)
    curvature = np.abs(np.gradient(np.gradient(pilot_y, unit), unit))
    density = curvature + 0.05 * max(float(np.max(curvature)), EPS)
    cumulative = np.cumsum(density)
    cumulative = (cumulative - cumulative[0]) / max(cumulative[-1] - cumulative[0], EPS)
    fits = []
    for count in (4, 8, 12, 18):
        quantiles = np.linspace(0.0, 1.0, count + 2)[1:-1]
        knots = np.unique(np.interp(quantiles, cumulative, unit))
        design = bspline_design(unit, len(knots) + 4, interior_knots=knots)
        penalty = 1e-8 * np.eye(design.shape[1])
        score, coefficients = _gcv(design, y, penalty)
        fits.append((score, knots, coefficients))
    _, knots, coefficients = min(fits, key=lambda item: item[0])

    def predict(new_x):
        new_unit = (np.asarray(new_x) - low) / width
        return bspline_design(
            new_unit, len(knots) + 4, interior_knots=knots
        ) @ coefficients

    return predict


def _matern_gp(x, y):
    unit, low, width = _scale_x(x)
    distance = np.abs(unit[:, None] - unit[None, :])
    best = None
    for length in (0.03, 0.06, 0.12, 0.24, 0.48):
        scaled = np.sqrt(3.0) * distance / length
        kernel = (1.0 + scaled) * np.exp(-scaled)
        for noise in (1e-7, 1e-5, 1e-3, 1e-1):
            inverse = np.linalg.inv(kernel + noise * np.eye(len(x)))
            alpha = inverse @ y
            fitted = kernel @ alpha
            degrees = float(np.trace(kernel @ inverse))
            denominator = max(1.0 - degrees / len(x), 1e-6)
            score = float(np.mean((y - fitted) ** 2) / denominator**2)
            candidate = (score, length, noise, alpha)
            best = candidate if best is None or score < best[0] else best
    _, length, _, alpha = best

    def predict(new_x):
        new_unit = np.clip((np.asarray(new_x) - low) / width, 0.0, 1.0)
        scaled = np.sqrt(3.0) * np.abs(new_unit[:, None] - unit[None, :]) / length
        return ((1.0 + scaled) * np.exp(-scaled)) @ alpha

    return predict


def _loess_predict(x, y, span):
    count = max(4, int(np.ceil(span * len(x))))

    def predict(new_x):
        output = np.empty_like(np.asarray(new_x, dtype=float))
        for output_index, point in enumerate(np.asarray(new_x, dtype=float)):
            distance = np.abs(x - point)
            bandwidth = max(float(np.partition(distance, count - 1)[count - 1]), EPS)
            ratio = np.minimum(distance / bandwidth, 1.0)
            weight = (1.0 - ratio**3) ** 3
            centered = x - point
            design = np.column_stack((np.ones(len(x)), centered, centered * centered))
            normal = design.T @ (weight[:, None] * design)
            rhs = design.T @ (weight * y)
            output[output_index] = np.linalg.solve(
                normal + 1e-10 * np.eye(3), rhs
            )[0]
        return output

    return predict


def _loess(x, y):
    folds = np.arange(len(x)) % 5
    candidates = (0.12, 0.20, 0.32, 0.48)
    scores = []
    for span in candidates:
        errors = []
        for held_out in range(5):
            train = folds != held_out
            curve = _loess_predict(x[train], y[train], span)
            errors.extend((curve(x[~train]) - y[~train]) ** 2)
        scores.append(float(np.mean(errors)))
    return _loess_predict(x, y, candidates[int(np.argmin(scores))])


def _chebyshev(x, y):
    unit, low, width = _scale_x(x)
    scaled = 2.0 * unit - 1.0
    fits = []
    for degree in (7, 11, 15, 19):
        degree = min(degree, len(x) - 2)
        design = chebvander(scaled, degree)
        penalty = 1e-8 * np.eye(degree + 1)
        penalty[0, 0] = 0.0
        score, coefficients = _gcv(design, y, penalty)
        fits.append((score, degree, coefficients))
    _, degree, coefficients = min(fits, key=lambda item: item[0])

    def predict(new_x):
        new_scaled = 2.0 * np.clip((np.asarray(new_x) - low) / width, 0.0, 1.0) - 1.0
        return chebvander(new_scaled, degree) @ coefficients

    return predict


def _piecewise_chebyshev(x, y):
    unit, low, width = _scale_x(x)
    centers = np.array([0.0, 0.5, 1.0])
    radii = np.array([0.58, 0.38, 0.58])
    fits = []
    for center, radius in zip(centers, radii):
        selected = np.abs(unit - center) <= radius
        local_x = unit[selected]
        local_y = y[selected]
        local_low, local_high = float(local_x[0]), float(local_x[-1])
        degree = min(9, len(local_x) - 2)
        design = chebvander(
            2.0 * (local_x - local_low) / (local_high - local_low) - 1.0,
            degree,
        )
        penalty = 1e-8 * np.eye(degree + 1)
        penalty[0, 0] = 0.0
        coefficients = _solve(design, local_y, penalty)
        fits.append((local_low, local_high, degree, coefficients))

    def predict(new_x):
        new_unit = np.clip((np.asarray(new_x) - low) / width, 0.0, 1.0)
        values = []
        weights = []
        for center, radius, (local_low, local_high, degree, coefficients) in zip(
            centers, radii, fits
        ):
            local_scaled = (
                2.0
                * (np.clip(new_unit, local_low, local_high) - local_low)
                / (local_high - local_low)
                - 1.0
            )
            values.append(chebvander(local_scaled, degree) @ coefficients)
            weights.append(np.maximum(1.0 - np.abs(new_unit - center) / radius, 0.0))
        weights = np.asarray(weights)
        weights /= np.maximum(np.sum(weights, axis=0), EPS)
        return np.sum(weights * np.asarray(values), axis=0)

    return predict


def _floater_hormann(x, y, degree=3):
    unit, low, width = _scale_x(x)
    n = len(unit) - 1
    degree = min(degree, n)

    def predict(new_x):
        new_unit = np.clip((np.asarray(new_x) - low) / width, 0.0, 1.0)
        output = np.empty_like(new_unit)
        for index, point in enumerate(new_unit):
            distance = point - unit
            exact = np.flatnonzero(np.abs(distance) <= 1e-14)
            if len(exact):
                output[index] = y[exact[0]]
                continue
            log_weight = np.empty(n - degree + 1)
            sign = np.empty(n - degree + 1)
            local_value = np.empty(n - degree + 1)
            for start in range(n - degree + 1):
                local_distance = distance[start : start + degree + 1]
                log_weight[start] = -np.sum(np.log(np.abs(local_distance)))
                sign[start] = (-1.0) ** start * np.prod(np.sign(local_distance))
                local_x = unit[start : start + degree + 1]
                local_y = y[start : start + degree + 1]
                local_weights = np.ones(degree + 1)
                for node in range(degree + 1):
                    local_weights[node] = 1.0 / np.prod(
                        local_x[node] - np.delete(local_x, node)
                    )
                ratio = local_weights / (point - local_x)
                local_value[start] = np.sum(ratio * local_y) / np.sum(ratio)
            scaled_weight = sign * np.exp(log_weight - np.max(log_weight))
            output[index] = np.sum(scaled_weight * local_value) / np.sum(scaled_weight)
        return output

    return predict


def _bernstein(x, y):
    unit, low, width = _scale_x(x)
    fits = []
    for degree in (7, 11, 15, 19):
        degree = min(degree, len(x) - 2)
        design = np.column_stack(
            [
                comb(degree, index)
                * unit**index
                * (1.0 - unit) ** (degree - index)
                for index in range(degree + 1)
            ]
        )
        penalty = 1e-8 * np.eye(degree + 1)
        score, coefficients = _gcv(design, y, penalty)
        fits.append((score, degree, coefficients))
    _, degree, coefficients = min(fits, key=lambda item: item[0])

    def predict(new_x):
        new_unit = np.clip((np.asarray(new_x) - low) / width, 0.0, 1.0)
        return np.column_stack(
            [
                comb(degree, index)
                * new_unit**index
                * (1.0 - new_unit) ** (degree - index)
                for index in range(degree + 1)
            ]
        ) @ coefficients

    return predict


METHODS = [
    "linear",
    "pchip",
    "akima",
    "makima",
    "natural_cubic_interpolation",
    "bspline_regression",
    "natural_cubic_regression",
    "cubic_smoothing_spline",
    "pspline",
    "adaptive_pspline",
    "free_knot_spline",
    "loess",
    "matern_gp",
    "chebyshev",
    "piecewise_chebyshev",
    "bernstein",
    "floater_hormann",
]


def fit_curve(x, y, method):
    x, y = _ordered_unique(x, y)
    if len(x) == 1:
        return lambda new_x: np.full_like(np.asarray(new_x), y[0], dtype=float)
    if method == "linear":
        return lambda new_x: np.interp(np.asarray(new_x), x, y)
    if method == "pchip":
        return hermite_predictor(x, y, pchip_slopes(x, y))
    if method == "akima":
        return hermite_predictor(x, y, akima_slopes(x, y))
    if method == "makima":
        return hermite_predictor(x, y, makima_slopes(x, y))
    if method == "natural_cubic_interpolation":
        return _natural_spline(x, y, 0.0)
    if method == "bspline_regression":
        return _linear_smoother(x, y, method)
    if method == "natural_cubic_regression":
        return _natural_regression(x, y)
    if method == "cubic_smoothing_spline":
        return _cv_natural_smoothing(x, y)
    if method in {"pspline", "adaptive_pspline"}:
        return _linear_smoother(x, y, method)
    if method == "free_knot_spline":
        return _adaptive_knot_regression(x, y)
    if method == "loess":
        return _loess(x, y)
    if method == "matern_gp":
        return _matern_gp(x, y)
    if method == "chebyshev":
        return _chebyshev(x, y)
    if method == "piecewise_chebyshev":
        return _piecewise_chebyshev(x, y)
    if method == "bernstein":
        return _bernstein(x, y)
    if method == "floater_hormann":
        return _floater_hormann(x, y)
    raise ValueError(method)
