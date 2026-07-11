"""Literature-inspired residual surrogates for the 7D SLV basket cliquet."""

import csv
import sys
from collections import defaultdict
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BasketCliquetOptExperiment.BasketCliquetMain as basket


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TRAIN_2001 = RESULTS / "basket_slv_cliquet_training_labels.csv"
TRAIN_5001 = RESULTS / "basket_slv_cliquet_training_labels_5001.csv"
DEV_DETAILS = RESULTS / "basket_slv_cliquet_proxy_validation_details.csv"
INDEPENDENT_DETAILS = (
    RESULTS / "basket_slv_cliquet_independent_validation_details.csv"
)
OUTPUT = RESULTS / "literature_surrogate_results.csv"
SUMMARY = (
    ROOT.parent
    / "Markdown"
    / "BasketCliquet"
    / "results"
    / "literature_surrogate_summary.md"
)
METHODS = [
    "normal_moment_baseline",
    "residual_hermite_2001",
    "residual_hermite_5001",
    "residual_local_5001",
    "residual_nystrom_5001",
    "residual_ensemble_5001",
    "direct_hermite_2001",
    "direct_hermite_5001",
    "direct_local_5001",
    "direct_nystrom_5001",
    "direct_ensemble_5001",
    "weighted_sparse_chebyshev_2001",
    "weighted_sparse_chebyshev_5001",
    "sparse_chebyshev_2001",
]
LOGIT_EPS = 1e-6


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normal_cdf(value):
    value = np.asarray(value, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(value / sqrt(2.0)))


def normal_call(mean, stdev, strike):
    stdev = np.maximum(stdev, 1e-10)
    d = (mean - strike) / stdev
    return (mean - strike) * normal_cdf(d) + stdev * np.exp(
        -0.5 * d * d
    ) / sqrt(2.0 * pi)


def moment_baseline(accrued, mean, variance, month, params):
    periods = basket.remaining_periods(month, params)
    total_mean = accrued + periods * mean
    total_stdev = np.sqrt(np.maximum(periods * variance, 1e-10))
    expected_clip = (
        params.global_floor
        + normal_call(total_mean, total_stdev, params.global_floor)
        - normal_call(total_mean, total_stdev, params.global_cap)
    )
    return (
        basket.discount(month, params)
        * params.notional
        * np.clip(expected_clip, params.global_floor, params.global_cap)
    )


def exact_values(accrued, month, params):
    return basket.exact_tail(accrued, month, params)


def arrays_from_training(rows):
    return {
        "accrued": np.array([float(row["accrued_return"]) for row in rows]),
        "spots": np.array(
            [[float(row[f"spot_{index}"]) for index in (1, 2, 3)] for row in rows]
        ),
        "variances": np.array(
            [
                [float(row[f"variance_{index}"]) for index in (1, 2, 3)]
                for row in rows
            ]
        ),
        "mean": np.array([float(row["coupon_mean"]) for row in rows]),
        "coupon_variance": np.array(
            [float(row["coupon_variance"]) for row in rows]
        ),
        "skewness": np.array([float(row["coupon_skewness"]) for row in rows]),
        "floor_mass": np.array([float(row["coupon_floor_mass"]) for row in rows]),
        "cap_mass": np.array([float(row["coupon_cap_mass"]) for row in rows]),
        "value": np.array([float(row["value"]) for row in rows]),
        "stderr": np.array([float(row["stderr"]) for row in rows]),
    }


def feature_matrix(data, month, params):
    periods = basket.remaining_periods(month, params)
    total_mean = data["accrued"] + periods * data["mean"]
    total_stdev = np.sqrt(
        np.maximum(periods * data["coupon_variance"], 1e-10)
    )
    lower = (total_mean - params.global_floor) / total_stdev
    upper = (params.global_cap - total_mean) / total_stdev
    log_spots = np.log(data["spots"] / np.asarray(params.s0))
    log_vars = np.log(
        np.maximum(data["variances"], 1e-10) / np.asarray(params.theta)
    )
    return np.column_stack(
        (
            lower,
            upper,
            data["skewness"],
            data["floor_mass"],
            data["cap_mass"],
            log_spots,
            log_vars,
            np.mean(log_spots, axis=1),
            np.std(log_spots, axis=1),
            np.min(log_spots, axis=1),
            np.max(log_spots, axis=1),
            np.mean(log_vars, axis=1),
            np.std(log_vars, axis=1),
            np.min(log_vars, axis=1),
            np.max(log_vars, axis=1),
        )
    )


def raw_feature_matrix(data, month, params):
    periods = basket.remaining_periods(month, params)
    total_mean = data["accrued"] + periods * data["mean"]
    total_stdev = np.sqrt(
        np.maximum(periods * data["coupon_variance"], 1e-10)
    )
    return np.column_stack(
        (
            (total_mean - params.global_floor) / total_stdev,
            (params.global_cap - total_mean) / total_stdev,
            np.log(data["spots"] / np.asarray(params.s0)),
            np.log(
                np.maximum(data["variances"], 1e-10)
                / np.asarray(params.theta)
            ),
        )
    )


def combine_training(primary, secondary):
    combined = {name: value.copy() for name, value in secondary.items()}
    count = len(primary["value"])
    variance_1 = np.maximum(primary["stderr"] ** 2, 1e-10)
    variance_2 = np.maximum(secondary["stderr"][:count] ** 2, 1e-10)
    weight_1 = 1.0 / variance_1
    weight_2 = 1.0 / variance_2
    combined["value"][:count] = (
        weight_1 * primary["value"] + weight_2 * secondary["value"][:count]
    ) / (weight_1 + weight_2)
    combined["stderr"][:count] = np.sqrt(1.0 / (weight_1 + weight_2))
    return combined


def validation_arrays(rows, variant, month, params):
    selected = [
        row
        for row in rows
        if row["variant"] == variant
        and int(row["month"]) == month
        and (
            "method" not in row
            or row["method"] == "sparse_chebyshev"
        )
    ]
    unique = {}
    for row in selected:
        key = int(row.get("state_index", len(unique)))
        unique.setdefault(key, row)
    selected = [unique[key] for key in sorted(unique)]
    spots = np.array(
        [[float(row[f"spot_{index}"]) for index in (1, 2, 3)] for row in selected]
    )
    variances = np.array(
        [
            [float(row[f"variance_{index}"]) for index in (1, 2, 3)]
            for row in selected
        ]
    )
    spot_z = basket.feature_normals(params)
    moments = [
        basket.frozen_coupon_moments(spot, variance, params, spot_z)[variant]
        for spot, variance in zip(spots, variances)
    ]
    return {
        "accrued": np.array([float(row["accrued_return"]) for row in selected]),
        "spots": spots,
        "variances": variances,
        "mean": np.array([item[0] for item in moments]),
        "coupon_variance": np.array([item[1] for item in moments]),
        "skewness": np.array([item[2] for item in moments]),
        "floor_mass": np.array([item[3] for item in moments]),
        "cap_mass": np.array([item[4] for item in moments]),
        "benchmark": np.array([float(row["benchmark"]) for row in selected]),
        "benchmark_stderr": np.array(
            [float(row["benchmark_stderr"]) for row in selected]
        ),
        "sparse_proxy": np.array([float(row["proxy"]) for row in selected]),
    }


def standardize(train_x):
    center = np.mean(train_x, axis=0)
    scale = np.maximum(np.std(train_x, axis=0), 1e-8)
    return (train_x - center) / scale, center, scale


def apply_standardize(values, center, scale):
    return np.clip((values - center) / scale, -6.0, 6.0)


def hermite_design(features):
    columns = [np.ones(len(features))]
    columns.extend(features[:, index] for index in range(features.shape[1]))
    columns.extend(
        (features[:, index] ** 2 - 1.0) / sqrt(2.0)
        for index in range(features.shape[1])
    )
    columns.extend(
        (features[:, index] ** 3 - 3.0 * features[:, index]) / sqrt(6.0)
        for index in range(features.shape[1])
    )
    for left in range(features.shape[1]):
        for right in range(left + 1, features.shape[1]):
            columns.append(features[:, left] * features[:, right])
    for cushion in (0, 1):
        h2 = (features[:, cushion] ** 2 - 1.0) / sqrt(2.0)
        for other in range(2, features.shape[1]):
            columns.append(h2 * features[:, other])
    return np.column_stack(columns)


def weighted_ridge(design, target, stderr, ridge):
    typical = max(float(np.median(stderr)), 1e-6)
    weight = np.minimum((typical / np.maximum(stderr, typical)) ** 2, 1.0)
    root = np.sqrt(weight / np.mean(weight))
    weighted = design * root[:, None]
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        weighted.T @ weighted + penalty,
        weighted.T @ (target * root),
    )


def fit_hermite(features, residual, stderr):
    scaled, center, scale = standardize(features)
    design = hermite_design(scaled)
    holdout = np.arange(len(features)) % 5 == 0
    candidates = []
    for ridge in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        coefficients = weighted_ridge(
            design[~holdout], residual[~holdout], stderr[~holdout], ridge
        )
        error = design[holdout] @ coefficients - residual[holdout]
        candidates.append((float(np.mean(error * error)), ridge))
    ridge = min(candidates)[1]
    coefficients = weighted_ridge(design, residual, stderr, ridge)

    def predict(new_features):
        return hermite_design(
            apply_standardize(new_features, center, scale)
        ) @ coefficients

    return predict


def fit_local(features, residual):
    scaled, center, scale = standardize(features)
    candidate_queries = np.arange(0, len(features), max(len(features) // 250, 1))
    candidate_queries = candidate_queries[:250]
    train_mask = np.ones(len(features), dtype=bool)
    train_mask[candidate_queries] = False

    def local_predict(query, count, source_x, source_y):
        output = np.empty(len(query))
        for row, point in enumerate(query):
            distance_sq = np.sum((source_x - point) ** 2, axis=1)
            neighbors = np.argpartition(distance_sq, count - 1)[:count]
            bandwidth = max(float(np.max(distance_sq[neighbors])), 1e-8)
            weight = np.exp(-3.0 * distance_sq[neighbors] / bandwidth)
            output[row] = np.sum(weight * source_y[neighbors]) / np.sum(weight)
        return output

    candidates = []
    for count in (64, 128, 256, 512):
        prediction = local_predict(
            scaled[candidate_queries],
            min(count, np.sum(train_mask)),
            scaled[train_mask],
            residual[train_mask],
        )
        candidates.append(
            (
                float(
                    np.mean(
                        (prediction - residual[candidate_queries]) ** 2
                    )
                ),
                count,
            )
        )
    count = min(candidates)[1]

    def predict(new_features):
        return local_predict(
            apply_standardize(new_features, center, scale),
            min(count, len(scaled)),
            scaled,
            residual,
        )

    return predict


def matern_kernel(left, right, length):
    left_sq = np.sum(left * left, axis=1)[:, None]
    right_sq = np.sum(right * right, axis=1)[None, :]
    distance = np.sqrt(
        np.maximum(left_sq + right_sq - 2.0 * left @ right.T, 0.0)
    )
    scaled = sqrt(3.0) * distance / length
    return (1.0 + scaled) * np.exp(-scaled)


def farthest_indices(features, count):
    selected = [0]
    minimum = np.sum((features - features[0]) ** 2, axis=1)
    for _ in range(1, min(count, len(features))):
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(
            minimum, np.sum((features - features[index]) ** 2, axis=1)
        )
    return np.array(selected)


def fit_nystrom(features, residual, stderr):
    scaled, center, scale = standardize(features)
    inducing = scaled[farthest_indices(scaled, 192)]
    holdout = np.arange(len(features)) % 5 == 0
    candidates = []
    cached = {}
    for length in (2.0, 3.5, 5.0, 7.0):
        k_mm = matern_kernel(inducing, inducing, length)
        eigenvalue, eigenvector = np.linalg.eigh(
            k_mm + 1e-8 * np.eye(len(inducing))
        )
        inverse_root = eigenvector @ np.diag(
            1.0 / np.sqrt(np.maximum(eigenvalue, 1e-10))
        ) @ eigenvector.T
        phi = matern_kernel(scaled, inducing, length) @ inverse_root
        cached[length] = (phi, inverse_root)
        for ridge in (1e-4, 1e-3, 1e-2, 1e-1):
            coefficients = weighted_ridge(
                phi[~holdout], residual[~holdout], stderr[~holdout], ridge
            )
            error = phi[holdout] @ coefficients - residual[holdout]
            candidates.append((float(np.mean(error * error)), length, ridge))
    _, length, ridge = min(candidates)
    phi, inverse_root = cached[length]
    coefficients = weighted_ridge(phi, residual, stderr, ridge)

    def predict(new_features):
        new_scaled = apply_standardize(new_features, center, scale)
        phi_new = matern_kernel(new_scaled, inducing, length) @ inverse_root
        return phi_new @ coefficients

    return predict


def prepare_models(training, month, params):
    features = feature_matrix(training, month, params)
    baseline = moment_baseline(
        training["accrued"],
        training["mean"],
        training["coupon_variance"],
        month,
        params,
    )
    lower = basket.discount(month, params) * params.notional * params.global_floor
    upper = basket.discount(month, params) * params.notional * params.global_cap
    width = max(upper - lower, 1e-12)
    normalized_value = np.clip(
        (training["value"] - lower) / width, LOGIT_EPS, 1.0 - LOGIT_EPS
    )
    normalized_baseline = np.clip(
        (baseline - lower) / width, LOGIT_EPS, 1.0 - LOGIT_EPS
    )
    value_logit = np.log(normalized_value / (1.0 - normalized_value))
    baseline_logit = np.log(
        normalized_baseline / (1.0 - normalized_baseline)
    )
    residual = value_logit - baseline_logit
    logit_stderr = training["stderr"] / np.maximum(
        width * normalized_value * (1.0 - normalized_value), 1e-4
    )
    return {
        "hermite": fit_hermite(features, residual, logit_stderr),
        "local": fit_local(features, residual),
        "nystrom": fit_nystrom(
            features, residual, logit_stderr
        ),
    }


def corrected_prediction(baseline, correction, month, params):
    lower = basket.discount(month, params) * params.notional * params.global_floor
    upper = basket.discount(month, params) * params.notional * params.global_cap
    width = max(upper - lower, 1e-12)
    normalized = np.clip(
        (baseline - lower) / width, LOGIT_EPS, 1.0 - LOGIT_EPS
    )
    baseline_logit = np.log(normalized / (1.0 - normalized))
    adjusted = baseline_logit + np.clip(correction, -12.0, 12.0)
    probability = 1.0 / (1.0 + np.exp(-np.clip(adjusted, -35.0, 35.0)))
    return lower + width * probability


def prepare_direct_models(training, month, params):
    features = feature_matrix(training, month, params)
    lower = basket.discount(month, params) * params.notional * params.global_floor
    upper = basket.discount(month, params) * params.notional * params.global_cap
    width = max(upper - lower, 1e-12)
    normalized = np.clip(
        (training["value"] - lower) / width, LOGIT_EPS, 1.0 - LOGIT_EPS
    )
    target = np.log(normalized / (1.0 - normalized))
    logit_stderr = training["stderr"] / np.maximum(
        width * normalized * (1.0 - normalized), 1e-4
    )
    return {
        "hermite": fit_hermite(features, target, logit_stderr),
        "local": fit_local(features, target),
        "nystrom": fit_nystrom(features, target, logit_stderr),
    }


def direct_prediction(raw, month, params):
    lower = basket.discount(month, params) * params.notional * params.global_floor
    upper = basket.discount(month, params) * params.notional * params.global_cap
    probability = 1.0 / (1.0 + np.exp(-np.clip(raw, -35.0, 35.0)))
    return lower + (upper - lower) * probability


def fit_weighted_sparse_chebyshev(training, month, params):
    features = raw_feature_matrix(training, month, params)
    scaled, low, high = basket.scale_features(features)
    design = basket.sparse_chebyshev_design(scaled)
    lower = basket.discount(month, params) * params.notional * params.global_floor
    upper = basket.discount(month, params) * params.notional * params.global_cap
    width = max(upper - lower, 1e-12)
    normalized = np.clip(
        (training["value"] - lower) / width, LOGIT_EPS, 1.0 - LOGIT_EPS
    )
    target = np.log(normalized / (1.0 - normalized))
    logit_stderr = training["stderr"] / np.maximum(
        width * normalized * (1.0 - normalized), 1e-4
    )
    holdout = np.arange(len(target)) % 5 == 0
    candidates = []
    for ridge in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        coefficients = weighted_ridge(
            design[~holdout],
            target[~holdout],
            logit_stderr[~holdout],
            ridge,
        )
        error = design[holdout] @ coefficients - target[holdout]
        candidates.append((float(np.mean(error * error)), ridge))
    ridge = min(candidates)[1]
    coefficients = weighted_ridge(
        design, target, logit_stderr, ridge
    )

    def predict(new_data):
        new_features = raw_feature_matrix(new_data, month, params)
        query = basket.apply_feature_scale(new_features, low, high)
        return direct_prediction(
            basket.sparse_chebyshev_design(query) @ coefficients,
            month,
            params,
        )

    return predict


def apply_exact(prediction, data, month, params):
    exact = exact_values(data["accrued"], month, params)
    mask = np.isfinite(exact)
    prediction = np.asarray(prediction).copy()
    prediction[mask] = exact[mask]
    lower = basket.discount(month, params) * params.notional * params.global_floor
    upper = basket.discount(month, params) * params.notional * params.global_cap
    return np.clip(prediction, lower, upper)


def score(prediction, benchmark):
    absolute = np.abs(prediction - benchmark)
    relative = absolute / np.maximum(np.abs(benchmark), basket.RELATIVE_ERROR_FLOOR)
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
    params = basket.Params()
    rows_2001 = read_csv(TRAIN_2001)
    rows_5001 = read_csv(TRAIN_5001)
    dev_rows = read_csv(DEV_DETAILS)
    independent_rows = read_csv(INDEPENDENT_DETAILS)
    grouped_2001 = defaultdict(list)
    grouped_5001 = defaultdict(list)
    for row in rows_2001:
        grouped_2001[(row["variant"], int(row["month"]))].append(row)
    for row in rows_5001:
        grouped_5001[(row["variant"], int(row["month"]))].append(row)

    result_rows = []
    for variant in basket.VARIANTS:
        for month in basket.TEST_MONTHS:
            key = (variant, month)
            train_2001 = arrays_from_training(grouped_2001[key])
            train_5001 = combine_training(
                train_2001, arrays_from_training(grouped_5001[key])
            )
            models_2001 = prepare_models(train_2001, month, params)
            models_5001 = prepare_models(train_5001, month, params)
            direct_2001 = prepare_direct_models(train_2001, month, params)
            direct_5001 = prepare_direct_models(train_5001, month, params)
            weighted_sparse_2001 = fit_weighted_sparse_chebyshev(
                train_2001, month, params
            )
            weighted_sparse_5001 = fit_weighted_sparse_chebyshev(
                train_5001, month, params
            )
            for design, source in (
                ("development", dev_rows),
                ("independent", independent_rows),
            ):
                validation = validation_arrays(
                    source, variant, month, params
                )
                features = feature_matrix(validation, month, params)
                baseline = moment_baseline(
                    validation["accrued"],
                    validation["mean"],
                    validation["coupon_variance"],
                    month,
                    params,
                )
                predictions = {
                    "normal_moment_baseline": baseline,
                    "residual_hermite_2001": corrected_prediction(
                        baseline,
                        models_2001["hermite"](features),
                        month,
                        params,
                    ),
                    "residual_hermite_5001": corrected_prediction(
                        baseline,
                        models_5001["hermite"](features),
                        month,
                        params,
                    ),
                    "residual_local_5001": corrected_prediction(
                        baseline,
                        models_5001["local"](features),
                        month,
                        params,
                    ),
                    "residual_nystrom_5001": corrected_prediction(
                        baseline,
                        models_5001["nystrom"](features),
                        month,
                        params,
                    ),
                    "direct_hermite_2001": direct_prediction(
                        direct_2001["hermite"](features), month, params
                    ),
                    "direct_hermite_5001": direct_prediction(
                        direct_5001["hermite"](features), month, params
                    ),
                    "direct_local_5001": direct_prediction(
                        direct_5001["local"](features), month, params
                    ),
                    "direct_nystrom_5001": direct_prediction(
                        direct_5001["nystrom"](features), month, params
                    ),
                    "weighted_sparse_chebyshev_2001": weighted_sparse_2001(
                        validation
                    ),
                    "weighted_sparse_chebyshev_5001": weighted_sparse_5001(
                        validation
                    ),
                    "sparse_chebyshev_2001": validation["sparse_proxy"],
                }
                ensemble_correction = (
                    models_5001["hermite"](features)
                    + models_5001["local"](features)
                    + models_5001["nystrom"](features)
                ) / 3.0
                predictions["residual_ensemble_5001"] = corrected_prediction(
                    baseline, ensemble_correction, month, params
                )
                predictions["direct_ensemble_5001"] = direct_prediction(
                    (
                        direct_5001["hermite"](features)
                        + direct_5001["local"](features)
                        + direct_5001["nystrom"](features)
                    )
                    / 3.0,
                    month,
                    params,
                )
                for method, prediction in predictions.items():
                    prediction = apply_exact(
                        prediction, validation, month, params
                    )
                    result_rows.append(
                        {
                            "variant": variant,
                            "month": month,
                            "validation_design": design,
                            "method": method,
                            **score(prediction, validation["benchmark"]),
                        }
                    )
            print(f"finished {variant}, month {month}", flush=True)

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    lines = [
        "# Literature-inspired SLV basket cliquet surrogates",
        "",
        "| Variant | Method | Worst max | Average p99 | Average MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for variant in basket.VARIANTS:
        aggregate = []
        for method in METHODS:
            selected = [
                row
                for row in result_rows
                if row["variant"] == variant
                and row["method"] == method
                and row["month"] < params.n_periods
            ]
            aggregate.append(
                (
                    max(row["max_rel"] for row in selected),
                    method,
                    float(np.mean([row["p99_rel"] for row in selected])),
                    float(np.mean([row["mae"] for row in selected])),
                )
            )
        for worst, method, average_p99, average_mae in sorted(aggregate):
            lines.append(
                f"| `{variant}` | `{method}` | {100 * worst:.3f}% | "
                f"{100 * average_p99:.3f}% | {average_mae:.6f} |"
            )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
