import csv
import sys
from math import comb
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import AsianMain as asian
import EuroMain as euro


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
RESULT_CSV = OUTPUT_DIR / "one_dimensional_method_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"
METHODS = [
    "chebyshev",
    "pchip",
    "akima",
    "bezier_global",
    "bezier_pchip",
]
RIDGE = 1e-8


def endpoint_limited_slope(slope, secant):
    if slope * secant <= 0.0:
        return 0.0
    if abs(slope) > 3.0 * abs(secant):
        return 3.0 * secant
    return slope


def pchip_slopes(x, y):
    if len(x) == 2:
        return np.array([(y[1] - y[0]) / (x[1] - x[0])] * 2)
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    same_sign = delta[:-1] * delta[1:] > 0.0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    denominator = (
        w1 / np.where(delta[:-1] == 0.0, 1.0, delta[:-1])
        + w2 / np.where(delta[1:] == 0.0, 1.0, delta[1:])
    )
    interior_slopes = np.zeros_like(denominator)
    np.divide(
        w1 + w2,
        denominator,
        out=interior_slopes,
        where=np.abs(denominator) > 1e-14,
    )
    slopes[1:-1] = np.where(same_sign, interior_slopes, 0.0)
    slopes[0] = endpoint_limited_slope(
        ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1])
        / (h[0] + h[1]),
        delta[0],
    )
    slopes[-1] = endpoint_limited_slope(
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
    slopes = np.empty(len(y))
    for index in range(len(y)):
        left_far = extended[index]
        left = extended[index + 1]
        right = extended[index + 2]
        right_far = extended[index + 3]
        weight_left = abs(right_far - right)
        weight_right = abs(left - left_far)
        if weight_left + weight_right > 1e-14:
            slopes[index] = (
                weight_left * left + weight_right * right
            ) / (weight_left + weight_right)
        else:
            slopes[index] = 0.5 * (left + right)
    return slopes


def cubic_hermite(x, y, slopes):
    def predict(new_x):
        new_x = np.asarray(new_x)
        index = np.clip(np.searchsorted(x, new_x) - 1, 0, len(x) - 2)
        h = x[index + 1] - x[index]
        t = np.clip((new_x - x[index]) / h, 0.0, 1.0)
        return (
            (2 * t**3 - 3 * t**2 + 1) * y[index]
            + (t**3 - 2 * t**2 + t) * h * slopes[index]
            + (-2 * t**3 + 3 * t**2) * y[index + 1]
            + (t**3 - t**2) * h * slopes[index + 1]
        )

    return predict


def cubic_bezier_from_slopes(x, y, slopes):
    # B0=y_i, B1=y_i+h*d_i/3, B2=y_{i+1}-h*d_{i+1}/3, B3=y_{i+1}.
    def predict(new_x):
        new_x = np.asarray(new_x)
        index = np.clip(np.searchsorted(x, new_x) - 1, 0, len(x) - 2)
        h = x[index + 1] - x[index]
        t = np.clip((new_x - x[index]) / h, 0.0, 1.0)
        b0 = y[index]
        b1 = y[index] + h * slopes[index] / 3.0
        b2 = y[index + 1] - h * slopes[index + 1] / 3.0
        b3 = y[index + 1]
        return (
            (1.0 - t) ** 3 * b0
            + 3.0 * (1.0 - t) ** 2 * t * b1
            + 3.0 * (1.0 - t) * t**2 * b2
            + t**3 * b3
        )

    return predict


def bernstein_design(unit_x, degree):
    return np.column_stack(
        [
            comb(degree, index)
            * unit_x**index
            * (1.0 - unit_x) ** (degree - index)
            for index in range(degree + 1)
        ]
    )


def fit_curve(x, y, method):
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]
    unique_x, unique_index = np.unique(x, return_index=True)
    x, y = unique_x, y[unique_index]
    if len(x) == 1:
        return lambda new_x: np.full_like(np.asarray(new_x), y[0], dtype=float)
    low, high = float(x[0]), float(x[-1])

    if method == "chebyshev":
        degree = min(19, len(x) - 1)
        scaled = 2.0 * (x - low) / (high - low) - 1.0
        design = chebvander(scaled, degree)
        penalty = np.eye(degree + 1) * RIDGE
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ y
        )

        def predict(new_x):
            new_scaled = np.clip(
                2.0 * (np.asarray(new_x) - low) / (high - low) - 1.0,
                -1.0,
                1.0,
            )
            return chebvander(new_scaled, degree) @ coefficients

        return predict
    if method == "pchip":
        curve = PchipInterpolator(x, y, extrapolate=True)
        return lambda new_x: curve(np.clip(np.asarray(new_x), x[0], x[-1]))
    if method == "akima":
        curve = Akima1DInterpolator(x, y)
        return lambda new_x: curve(np.clip(np.asarray(new_x), x[0], x[-1]))
    if method == "bezier_pchip":
        return cubic_bezier_from_slopes(x, y, pchip_slopes(x, y))
    if method == "bezier_global":
        degree = min(12, len(x) - 1)
        unit = (x - low) / (high - low)
        design = bernstein_design(unit, degree)
        penalty = np.eye(degree + 1) * RIDGE
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ y
        )

        def predict(new_x):
            new_unit = np.clip(
                (np.asarray(new_x) - low) / (high - low), 0.0, 1.0
            )
            return bernstein_design(new_unit, degree) @ coefficients

        return predict
    raise ValueError(method)


def euro_experiment(rng):
    params = euro.Params()
    rows = []
    for fraction in euro.TIME_FRACTIONS:
        time = params.maturity * fraction
        tau = params.maturity - time
        validation_spot = euro.delta_space_spot_grid(tau, params, euro.GRID_POINTS)
        benchmark = euro.black_scholes_value(validation_spot, tau, params)
        if tau <= 0.0:
            predictions = {method: euro.payoff(validation_spot, params) for method in METHODS}
        else:
            train_spot = euro.delta_space_spot_grid(tau, params, euro.STATE_POINTS)
            train_value = euro.shifted_mc_value(train_spot, tau, params, rng)
            x = euro.d1_from_spot(train_spot, tau, params)
            validation_x = euro.d1_from_spot(validation_spot, tau, params)
            predictions = {}
            for method in METHODS:
                curve = fit_curve(
                    x, np.log(np.maximum(train_value, 0.0) + 1e-10), method
                )
                predictions[method] = np.maximum(
                    np.exp(np.clip(curve(validation_x), -30.0, 20.0)) - 1e-10,
                    0.0,
                )
        for method in METHODS:
            metrics = euro.score(predictions[method], benchmark)
            rows.append(
                {
                    "product": "european",
                    "method": method,
                    "time_index": fraction,
                    "remaining_steps": 0,
                    **metrics,
                }
            )
        print(f"finished European time {time:.2f}")
    return rows


def fit_asian_proxy(
    train_spot,
    train_sum,
    train_value,
    day_index,
    params,
    method,
):
    m = asian.future_count(day_index, params)
    if m == 0:
        return lambda spot, running: np.maximum(
            asian.linear_values(spot, running, day_index, params), 0.0
        )
    strike_adj = asian.adjusted_strike(
        train_spot, train_sum, day_index, params
    )
    active = strike_adj > 0.0
    x = asian.adjusted_moneyness_coordinate(
        train_spot[active], train_sum[active], day_index, params
    )
    base = np.maximum(
        asian.linear_values(
            train_spot[active], train_sum[active], day_index, params
        ),
        0.0,
    )
    normalized_value = np.maximum(
        train_value[active] / train_spot[active], 0.0
    )
    normalized_time = np.maximum(
        (train_value[active] - base) / train_spot[active], 0.0
    )
    value_curve = fit_curve(
        x, np.log(normalized_value + asian.LOG_EPS), method
    )
    time_curve = fit_curve(
        x, np.log(normalized_time + asian.LOG_EPS), method
    )

    def predict(spot, running):
        spot = np.asarray(spot)
        running = np.asarray(running)
        result = np.empty_like(spot)
        new_strike = asian.adjusted_strike(spot, running, day_index, params)
        linear_tail = new_strike <= 0.0
        result[linear_tail] = np.maximum(
            asian.linear_values(
                spot[linear_tail], running[linear_tail], day_index, params
            ),
            0.0,
        )
        fitted = ~linear_tail
        new_x = asian.adjusted_moneyness_coordinate(
            spot[fitted], running[fitted], day_index, params
        )
        base_value = np.maximum(
            asian.linear_values(
                spot[fitted], running[fitted], day_index, params
            ),
            0.0,
        )
        value_proxy = spot[fitted] * (
            np.exp(np.clip(value_curve(new_x), -30.0, 20.0)) - asian.LOG_EPS
        )
        time_proxy = base_value + spot[fitted] * (
            np.exp(np.clip(time_curve(new_x), -30.0, 20.0)) - asian.LOG_EPS
        )
        use_time = (
            base_value / np.maximum(spot[fitted], 1e-12)
        ) > asian.HYBRID_INTRINSIC_SWITCH
        result[fitted] = np.where(use_time, time_proxy, value_proxy)
        return np.maximum(result, 0.0)

    return predict


def asian_experiment(rng):
    params = asian.Params()
    rows = []
    for day_index in asian.TEST_DAY_INDICES:
        validation_spot, validation_sum = asian.make_state_grid(
            day_index,
            params,
            asian.VALIDATION_SPOT_POINTS,
            asian.VALIDATION_AVG_POINTS,
        )
        benchmark, benchmark_stderr = asian.build_labels(
            validation_spot,
            validation_sum,
            day_index,
            params,
            rng,
            asian.BENCHMARK_PATHS_PER_STATE,
        )
        train_spot, train_sum = asian.make_adjusted_moneyness_grid(
            day_index, params, asian.ADJUSTED_MONEYNESS_POINTS
        )
        paths = asian.paths_per_state_from_budget(
            asian.TRAIN_SCENARIOS_PER_FIT, len(train_spot)
        )
        train_value, _ = asian.build_labels(
            train_spot, train_sum, day_index, params, rng, paths
        )
        for method in METHODS:
            proxy = fit_asian_proxy(
                train_spot,
                train_sum,
                train_value,
                day_index,
                params,
                method,
            )
            prediction = proxy(validation_spot, validation_sum)
            metrics = asian.score(
                prediction, benchmark, benchmark_stderr
            )
            rows.append(
                {
                    "product": "asian",
                    "method": method,
                    "time_index": day_index,
                    "remaining_steps": asian.future_count(day_index, params),
                    "max_rel": metrics["max_rel"],
                    "p99_rel": metrics["p99_rel"],
                    "mae": metrics["mae"],
                    "max_abs": metrics["max_abs"],
                }
            )
        print(f"finished Asian fixing {day_index}")
    return rows


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = euro_experiment(np.random.default_rng(907))
    rows.extend(asian_experiment(np.random.default_rng(911)))
    with RESULT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# One-dimensional fitting comparison",
        "",
        "| Product | Method | Worst max relative error | Average p99 | Average MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for product in ("european", "asian"):
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["product"] == product and row["method"] == method
            ]
            lines.append(
                f"| {product} | `{method}` | "
                f"{100 * max(row['max_rel'] for row in selected):.3f}% | "
                f"{100 * np.mean([row['p99_rel'] for row in selected]):.3f}% | "
                f"{np.mean([row['mae'] for row in selected]):.6f} |"
            )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"results: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
