from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from compare_stage1_smoothers import (
    binned_series,
    bspline_design,
    fit_bspline,
    polynomial_design,
    stream_training_targets,
    weighted_quantile,
    weighted_ridge,
)
from stage1_lsmc_european import (
    GBMParams,
    asymptotic_anchor_value,
    asymptotic_three_region_weight,
    black_scholes_value,
    normalized_delta_for_cutoff,
    option_upper_bound,
    proxy_value,
)


CSV_PATH = Path("stage1_accuracy_experiment.csv")
RELATIVE_ERROR_FLOOR = 0.01


@dataclass(frozen=True)
class CandidateScore:
    method: str
    time: float
    mae: float
    rmse: float
    rel_mae: float
    max_abs_error: float
    detail: str


def clip_to_european_bounds(
    values: np.ndarray, spot: np.ndarray, tau: float, params: GBMParams
) -> np.ndarray:
    anchor = asymptotic_anchor_value(spot, tau, params)
    upper = option_upper_bound(spot, tau, params)
    return np.minimum(np.maximum(values, anchor), upper)


def three_region_values(
    spot: np.ndarray,
    tau: float,
    learned_values: np.ndarray,
    params: GBMParams,
    tail_delta: float,
    trained_delta: float,
) -> np.ndarray:
    anchor = asymptotic_anchor_value(spot, tau, params)
    delta_score = normalized_delta_for_cutoff(spot, tau, params)
    weight = asymptotic_three_region_weight(
        delta_score, tail_delta=tail_delta, trained_delta=trained_delta
    )
    mixed = weight * learned_values + (1.0 - weight) * anchor
    return clip_to_european_bounds(mixed, spot, tau, params)


def score_values(
    method: str,
    detail: str,
    time: float,
    prediction: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> CandidateScore:
    error = prediction - truth
    abs_error = np.abs(error)
    rel_error = abs_error / np.maximum(np.abs(truth), RELATIVE_ERROR_FLOOR)
    norm_weights = weights / weights.sum()
    return CandidateScore(
        method=method,
        time=time,
        mae=float(np.sum(norm_weights * abs_error)),
        rmse=float(np.sqrt(np.sum(norm_weights * error**2))),
        rel_mae=float(np.sum(norm_weights * rel_error)),
        max_abs_error=float(abs_error.max()),
        detail=detail,
    )


def best_three_region_score(
    method: str,
    spot: np.ndarray,
    tau: float,
    learned_values: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    params: GBMParams,
    time: float,
) -> CandidateScore:
    tail_grid = [1e-4, 3e-4, 1e-3, 3e-3]
    trained_grid = [0.003, 0.005, 0.01, 0.02, 0.05]
    best_score: CandidateScore | None = None

    for tail_delta in tail_grid:
        for trained_delta in trained_grid:
            if trained_delta <= tail_delta:
                continue
            prediction = three_region_values(
                spot, tau, learned_values, params, tail_delta, trained_delta
            )
            score = score_values(
                method,
                f"tail={tail_delta:g}, trained={trained_delta:g}",
                time,
                prediction,
                truth,
                weights,
            )
            if best_score is None or score.mae < best_score.mae:
                best_score = score

    if best_score is None:
        raise RuntimeError("empty threshold grid")
    return best_score


def weighted_poly_prediction(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    params: GBMParams,
    degree: int,
    ridge: float,
) -> np.ndarray:
    design = polynomial_design(spot, params, degree)
    coeffs = weighted_ridge(design, target, weights, ridge)
    return design @ coeffs


def log_poly_prediction(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    params: GBMParams,
    epsilon: float,
) -> np.ndarray:
    design = polynomial_design(spot, params, degree=9)
    coeffs = weighted_ridge(design, np.log(np.maximum(target, 0.0) + epsilon), weights, 1e-6)
    return np.exp(design @ coeffs) - epsilon


def residual_poly_prediction(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    tau: float,
    params: GBMParams,
) -> np.ndarray:
    anchor = asymptotic_anchor_value(spot, tau, params)
    residual_target = np.maximum(target - anchor, 0.0)
    design = polynomial_design(spot, params, degree=9)
    coeffs = weighted_ridge(design, residual_target, weights, 1e-6)
    return anchor + np.maximum(design @ coeffs, 0.0)


def residual_bspline_prediction(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    tau: float,
    params: GBMParams,
    n_internal_knots: int = 14,
    ridge: float = 1e-6,
) -> np.ndarray:
    anchor = asymptotic_anchor_value(spot, tau, params)
    residual_target = np.maximum(target - anchor, 0.0)
    z = spot / params.s0 - 1.0
    probabilities = np.linspace(0.0, 1.0, n_internal_knots + 2)[1:-1]
    internal = weighted_quantile(z, weights, probabilities)
    z_min = float(z.min())
    z_max = float(z.max())
    knots = np.concatenate(([z_min] * 4, internal, [z_max] * 4))
    design = bspline_design(z, knots)
    coeffs = weighted_ridge(design, residual_target, weights, ridge)
    return anchor + np.maximum(design @ coeffs, 0.0)


def piecewise_delta_poly_prediction(
    spot: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    tau: float,
    params: GBMParams,
) -> np.ndarray:
    delta_score = normalized_delta_for_cutoff(spot, tau, params)
    anchor = asymptotic_anchor_value(spot, tau, params)
    prediction = anchor.copy()

    regions = [
        (0.001, 0.15, 5),
        (0.15, 0.85, 7),
        (0.85, 0.999, 5),
    ]
    for low, high, degree in regions:
        mask = (delta_score >= low) & (delta_score <= high)
        if np.count_nonzero(mask) <= degree + 2:
            continue
        design = polynomial_design(spot[mask], params, degree)
        coeffs = weighted_ridge(design, target[mask], weights[mask], 1e-6)
        prediction[mask] = polynomial_design(spot[mask], params, degree) @ coeffs

    return prediction


def aggregate_scores(scores: list[CandidateScore]) -> list[CandidateScore]:
    methods = sorted({score.method for score in scores})
    aggregate = []
    for method in methods:
        method_scores = [score for score in scores if score.method == method]
        detail = " | ".join(sorted({score.detail for score in method_scores}))
        aggregate.append(
            CandidateScore(
                method=method,
                time=-1.0,
                mae=float(np.mean([score.mae for score in method_scores])),
                rmse=float(np.mean([score.rmse for score in method_scores])),
                rel_mae=float(np.mean([score.rel_mae for score in method_scores])),
                max_abs_error=float(max(score.max_abs_error for score in method_scores)),
                detail=detail,
            )
        )
    return sorted(aggregate, key=lambda score: score.mae)


def main() -> None:
    params = GBMParams(n_paths=10_000_000, n_steps=5, seed=7, option_type="call")
    start = perf_counter()
    (
        times,
        path_poly_coeffs_by_time,
        bin_edges_by_time,
        bin_sums_by_time,
        bin_counts_by_time,
    ) = stream_training_targets(params)

    scores: list[CandidateScore] = []

    for step in range(1, params.n_steps + 1):
        time = float(times[step])
        tau = float(params.maturity - times[step])
        spot, target, counts = binned_series(
            bin_edges_by_time[step], bin_sums_by_time[step], bin_counts_by_time[step]
        )
        weights = counts.astype(float)
        truth = black_scholes_value(spot, tau, params)

        learned_path_poly = proxy_value(spot, path_poly_coeffs_by_time[step], params)
        scores.append(
            score_values(
                "current_default_three_region_path_poly",
                "tail=0.001, trained=0.01",
                time,
                three_region_values(spot, tau, learned_path_poly, params, 0.001, 0.01),
                truth,
                weights,
            )
        )
        scores.append(
            best_three_region_score(
                "tuned_threshold_path_poly",
                spot,
                tau,
                learned_path_poly,
                truth,
                weights,
                params,
                time,
            )
        )

        candidate_predictions = {
            "binned_poly9": weighted_poly_prediction(
                spot, target, weights, params, degree=9, ridge=1e-6
            ),
            "weighted_relative_poly9": weighted_poly_prediction(
                spot,
                target,
                weights / np.sqrt(np.maximum(target, RELATIVE_ERROR_FLOOR)),
                params,
                degree=9,
                ridge=1e-6,
            ),
            "log_value_poly9_eps_0.01": log_poly_prediction(
                spot, target, weights, params, epsilon=0.01
            ),
            "time_value_residual_poly9": residual_poly_prediction(
                spot, target, weights, tau, params
            ),
            "binned_bspline14": fit_bspline(
                spot, target, weights, params, n_internal_knots=14, ridge=1e-6
            ).predict(spot),
            "time_value_residual_bspline14": residual_bspline_prediction(
                spot, target, weights, tau, params
            ),
            "piecewise_delta_poly": piecewise_delta_poly_prediction(
                spot, target, weights, tau, params
            ),
        }

        for method, learned_values in candidate_predictions.items():
            learned_values = clip_to_european_bounds(learned_values, spot, tau, params)
            scores.append(
                score_values(
                    f"{method}_default_thresholds",
                    "tail=0.001, trained=0.01",
                    time,
                    three_region_values(spot, tau, learned_values, params, 0.001, 0.01),
                    truth,
                    weights,
                )
            )
            scores.append(
                best_three_region_score(
                    f"{method}_tuned_thresholds",
                    spot,
                    tau,
                    learned_values,
                    truth,
                    weights,
                    params,
                    time,
                )
            )

    aggregate = aggregate_scores(scores)
    baseline = next(
        score for score in aggregate if score.method == "current_default_three_region_path_poly"
    )

    with CSV_PATH.open("w", encoding="utf-8") as file:
        file.write("method,time,mae,rmse,rel_mae,max_abs_error,detail\n")
        for score in sorted(scores, key=lambda item: (item.time, item.mae)):
            file.write(
                f"{score.method},{score.time:.6f},{score.mae:.10f},"
                f"{score.rmse:.10f},{score.rel_mae:.10f},"
                f"{score.max_abs_error:.10f},{score.detail}\n"
            )
        file.write("\n")
        file.write("overall_method,mae,rmse,rel_mae,max_abs_error,improvement_vs_baseline,detail\n")
        for score in aggregate:
            improvement = (baseline.mae - score.mae) / baseline.mae
            file.write(
                f"{score.method},{score.mae:.10f},{score.rmse:.10f},"
                f"{score.rel_mae:.10f},{score.max_abs_error:.10f},"
                f"{improvement:.10f},{score.detail}\n"
            )

    print()
    print("Stage 1 accuracy experiment")
    print("fixed params: European call, S0=100, K=100, r=5%, q=2%, vol=20%, T=1")
    print("training target: 10M-path binned discounted payoff CE")
    print("scoring target: closed-form Black-Scholes on weighted spot bins")
    print()
    print("overall ranking by average weighted MAE")
    print("rank  method                                      MAE       RMSE      rel MAE   improvement")
    print("----  ------                                      ---       ----      -------   -----------")
    for rank, score in enumerate(aggregate, start=1):
        improvement = (baseline.mae - score.mae) / baseline.mae
        print(
            f"{rank:4d}  {score.method:<42} "
            f"{score.mae:8.5f}  {score.rmse:8.5f}  "
            f"{100.0 * score.rel_mae:8.2f}%  {100.0 * improvement:10.2f}%"
        )

    print()
    print("best threshold choices for tuned candidates are in the CSV.")
    print(f"table written to: {CSV_PATH.resolve()}")
    print(f"elapsed seconds: {perf_counter() - start:.1f}")


if __name__ == "__main__":
    main()
