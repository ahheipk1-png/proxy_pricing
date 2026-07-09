"""Independent two-design study of 3D single-name SLV cliquet surrogates."""

import csv
import sys
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import SLVCliquetOptExperiment.SLVCliquetMain as slv
from BasketCliquetOptExperiment.LiteratureSurrogateStudy import (
    fit_hermite,
    fit_local,
    fit_nystrom,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "literature_surrogate_results.csv"
SUMMARY = ROOT / "results" / "literature_surrogate_summary.md"
DEGREES = (13, 15, 17, 19, 21, 23)
LOGIT_EPS = 1e-7


def normal_cdf(value):
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(value) / sqrt(2.0)))


def normal_call(mean, stdev, strike):
    stdev = np.maximum(stdev, 1e-10)
    d = (mean - strike) / stdev
    return (mean - strike) * normal_cdf(d) + stdev * np.exp(
        -0.5 * d * d
    ) / sqrt(2.0 * pi)


def normal_baseline(states, day_index, params):
    accrued, spot, variance = states
    periods = slv.remaining_periods(day_index, params)
    mean, coupon_variance = slv.frozen_coupon_moments(
        spot, variance, params, draws=2001
    )
    total_mean = accrued + periods * mean
    total_stdev = np.sqrt(np.maximum(periods * coupon_variance, 1e-10))
    clipped = (
        params.global_floor
        + normal_call(total_mean, total_stdev, params.global_floor)
        - normal_call(total_mean, total_stdev, params.global_cap)
    )
    return (
        slv.discount(day_index, params)
        * params.notional
        * np.clip(clipped, params.global_floor, params.global_cap)
    )


def target_logit(values, day_index, params):
    lower = slv.discount(day_index, params) * params.notional * params.global_floor
    upper = slv.discount(day_index, params) * params.notional * params.global_cap
    normalized = np.clip(
        (values - lower) / max(upper - lower, 1e-12),
        LOGIT_EPS,
        1.0 - LOGIT_EPS,
    )
    return np.log(normalized / (1.0 - normalized)), normalized


def from_logit(raw, day_index, params):
    lower = slv.discount(day_index, params) * params.notional * params.global_floor
    upper = slv.discount(day_index, params) * params.notional * params.global_cap
    probability = 1.0 / (1.0 + np.exp(-np.clip(raw, -35.0, 35.0)))
    return lower + (upper - lower) * probability


def feature_values(states, day_index, params):
    return np.column_stack(
        slv.structured_features(*states, day_index, params)
    )


def fit_anisotropic(states, values, day_index, params, degree):
    accrued, _, _ = states
    exact = slv.exact_tail_value(accrued, day_index, params)
    active = ~np.isfinite(exact)
    z, spot, variance = slv.structured_features(
        *(item[active] for item in states), day_index, params
    )
    z_scaled, z_low, z_high = slv.scale(z)
    spot_scaled, spot_low, spot_high = slv.scale(spot)
    var_scaled, var_low, var_high = slv.scale(variance)
    target, _ = target_logit(values[active], day_index, params)
    design = slv.anisotropic_design(
        z_scaled, spot_scaled, var_scaled, degree
    )
    folds = np.arange(len(target)) % 5
    candidates = []
    for ridge in (1e-8, 3e-7, 1e-5, 3e-4):
        error = []
        for fold in range(5):
            train = folds != fold
            penalty = ridge * np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            coefficients = np.linalg.solve(
                design[train].T @ design[train] + penalty,
                design[train].T @ target[train],
            )
            error.extend(design[~train] @ coefficients - target[~train])
        candidates.append((float(np.mean(np.asarray(error) ** 2)), ridge))
    ridge = min(candidates)[1]
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ target
    )

    def predict(new_states):
        exact_new = slv.exact_tail_value(
            new_states[0], day_index, params
        )
        output = np.empty_like(new_states[0])
        tail = np.isfinite(exact_new)
        output[tail] = exact_new[tail]
        if np.any(~tail):
            new_z, new_spot, new_var = slv.structured_features(
                *(item[~tail] for item in new_states),
                day_index,
                params,
            )
            new_design = slv.anisotropic_design(
                slv.apply_scale(new_z, z_low, z_high),
                slv.apply_scale(new_spot, spot_low, spot_high),
                slv.apply_scale(new_var, var_low, var_high),
                degree,
            )
            output[~tail] = from_logit(
                new_design @ coefficients, day_index, params
            )
        return output

    return predict


def fit_generic(states, values, stderr, day_index, params, method):
    exact = slv.exact_tail_value(states[0], day_index, params)
    active = ~np.isfinite(exact)
    features = feature_values(
        tuple(item[active] for item in states), day_index, params
    )
    target, normalized = target_logit(values[active], day_index, params)
    width = (
        slv.discount(day_index, params)
        * params.notional
        * (params.global_cap - params.global_floor)
    )
    logit_stderr = stderr[active] / np.maximum(
        width * normalized * (1.0 - normalized), 1e-4
    )
    if method == "hermite":
        model = fit_hermite(features, target, logit_stderr)
    elif method == "local":
        model = fit_local(features, target)
    elif method == "nystrom":
        model = fit_nystrom(features, target, logit_stderr)
    else:
        raise ValueError(method)

    def predict(new_states):
        exact_new = slv.exact_tail_value(
            new_states[0], day_index, params
        )
        output = np.empty_like(new_states[0])
        tail = np.isfinite(exact_new)
        output[tail] = exact_new[tail]
        if np.any(~tail):
            new_features = feature_values(
                tuple(item[~tail] for item in new_states),
                day_index,
                params,
            )
            output[~tail] = from_logit(
                model(new_features), day_index, params
            )
        return output

    return predict


def metrics(prediction, benchmark):
    absolute = np.abs(prediction - benchmark)
    relative = absolute / np.maximum(
        np.abs(benchmark), slv.RELATIVE_ERROR_FLOOR
    )
    meaningful = np.abs(benchmark) >= 0.05
    return {
        "max_rel": float(np.max(relative)),
        "p99_rel": float(np.quantile(relative, 0.99)),
        "meaningful_max_rel": float(np.max(relative[meaningful]))
        if np.any(meaningful)
        else float(np.max(relative)),
        "mae": float(np.mean(absolute)),
    }


def run():
    params = slv.Params()
    rows = []
    for day_index in [
        value
        for value in slv.TEST_DAY_INDICES
        if value < params.n_periods
    ]:
        train_states = slv.make_states(
            day_index, params, slv.TRAIN_STATES
        )
        train_paths = int(
            np.ceil(slv.TRAIN_SCENARIOS_PER_FIT / slv.TRAIN_STATES)
        )
        train_value, train_stderr = slv.build_labels(
            train_states,
            day_index,
            params,
            np.random.default_rng(70_000 + day_index),
            train_paths,
        )
        models = {
            f"anisotropic_chebyshev_d{degree}": fit_anisotropic(
                train_states,
                train_value,
                day_index,
                params,
                degree,
            )
            for degree in DEGREES
        }
        models.update(
            {
                method: fit_generic(
                    train_states,
                    train_value,
                    train_stderr,
                    day_index,
                    params,
                    method,
                )
                for method in ("hermite", "local", "nystrom")
            }
        )
        models["adaptive_hybrid_current"] = slv.fit_proxy(
            train_states,
            train_value,
            day_index,
            params,
            "adaptive_hybrid",
        )
        for design, offset in (
            ("development", 5003),
            ("independent", 11003),
        ):
            validation_states = slv.make_states(
                day_index,
                params,
                slv.VALIDATION_STATES,
                validation=True,
                validation_offset=offset,
            )
            benchmark, benchmark_stderr = slv.build_labels(
                validation_states,
                day_index,
                params,
                np.random.default_rng(80_000 + offset + day_index),
                slv.BENCHMARK_PATHS_PER_STATE,
            )
            for method, model in models.items():
                prediction = model(validation_states)
                rows.append(
                    {
                        "day_index": day_index,
                        "validation_design": design,
                        "method": method,
                        **metrics(prediction, benchmark),
                        "avg_benchmark_stderr": float(
                            np.mean(benchmark_stderr)
                        ),
                    }
                )
        print(f"finished month {day_index}", flush=True)

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    methods = sorted(set(row["method"] for row in rows))
    lines = [
        "# Single-name SLV cliquet literature surrogate study",
        "",
        "| Method | Worst max | Average p99 | Average MAE |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        selected = [
            row
            for row in rows
            if row["method"] == method
            and row["day_index"] < params.n_periods
        ]
        lines.append(
            f"| `{method}` | {100 * max(row['max_rel'] for row in selected):.3f}% | "
            f"{100 * np.mean([row['p99_rel'] for row in selected]):.3f}% | "
            f"{np.mean([row['mae'] for row in selected]):.6f} |"
        )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
