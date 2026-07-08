import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from compare_stage1_smoothers import ModelFit
from plot_stage1_10m_proxy import RELATIVE_ERROR_FLOOR
from plot_stage1_mc_tail_biased_no_asym import (
    fit_bspline_model,
    fit_chebyshev_model,
    fit_polynomial,
)
from plot_stage1_mc_wing_shift_proxy import (
    MC_PATHS_PER_STATE,
    N_STATE_POINTS,
    delta_space_spot_grid,
    shifted_mc_option_value,
)
from stage1_lsmc_european import GBMParams, black_scholes_value


OUTPUT_DIR = Path("findings/european_option_grid")
CSV_PATH = OUTPUT_DIR / "european_option_grid_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"

OPTION_TYPES = ["call", "put"]
STRIKES = [80.0, 100.0, 120.0]
VOLS = [0.10, 0.20, 0.40]
MATURITIES = [0.5, 1.0, 2.0]
TIME_FRACTIONS = [0.2, 0.4, 0.6, 0.8]
GRID_POINTS = 501
BASE_SEED = 20260708


@dataclass(frozen=True)
class ResultRow:
    option_type: str
    strike: float
    vol: float
    maturity: float
    time: float
    tau: float
    method: str
    detail: str
    max_rel: float
    p99_rel: float
    mae: float
    max_abs: float


def build_candidate_models(
    spot: np.ndarray, target: np.ndarray, tau: float, params: GBMParams
) -> list[ModelFit]:
    return [
        fit_chebyshev_model(
            spot,
            target,
            tau,
            params,
            coord="d1",
            degree=7,
            target_kind="log",
        ),
        fit_bspline_model(
            spot,
            target,
            tau,
            params,
            coord="d1",
            n_knots=12,
            target_kind="log",
        ),
        fit_polynomial(
            spot,
            target,
            tau,
            params,
            coord="spot",
            degree=9,
            target_kind="log",
        ),
    ]


def score_model(
    model: ModelFit,
    tau: float,
    params: GBMParams,
) -> tuple[float, float, float, float]:
    grid_spot = delta_space_spot_grid(tau, params, GRID_POINTS)
    truth = black_scholes_value(grid_spot, tau, params)
    prediction = np.asarray(model.predict(grid_spot), dtype=float)
    error = prediction - truth
    abs_error = np.abs(error)
    rel_error = abs_error / np.maximum(np.abs(truth), RELATIVE_ERROR_FLOOR)
    return (
        float(rel_error.max()),
        float(np.quantile(rel_error, 0.99)),
        float(abs_error.mean()),
        float(abs_error.max()),
    )


def run_case(
    params: GBMParams,
    combo_index: int,
) -> list[ResultRow]:
    rows = []
    for step_index, fraction in enumerate(TIME_FRACTIONS, start=1):
        time = params.maturity * fraction
        tau = params.maturity - time
        state_spot = delta_space_spot_grid(tau, params, N_STATE_POINTS)
        rng = np.random.default_rng(BASE_SEED + combo_index * 100 + step_index)
        state_value = shifted_mc_option_value(
            state_spot,
            tau,
            params,
            rng,
            MC_PATHS_PER_STATE,
        )

        for model in build_candidate_models(state_spot, state_value, tau, params):
            max_rel, p99_rel, mae, max_abs = score_model(model, tau, params)
            rows.append(
                ResultRow(
                    option_type=params.option_type,
                    strike=params.strike,
                    vol=params.vol,
                    maturity=params.maturity,
                    time=time,
                    tau=tau,
                    method=model.name,
                    detail=model.detail,
                    max_rel=max_rel,
                    p99_rel=p99_rel,
                    mae=mae,
                    max_abs=max_abs,
                )
            )
    return rows


def write_csv(rows: list[ResultRow]) -> None:
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "option_type",
                "strike",
                "vol",
                "maturity",
                "time",
                "tau",
                "method",
                "detail",
                "max_rel",
                "p99_rel",
                "mae",
                "max_abs",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.option_type,
                    f"{row.strike:.6f}",
                    f"{row.vol:.6f}",
                    f"{row.maturity:.6f}",
                    f"{row.time:.6f}",
                    f"{row.tau:.6f}",
                    row.method,
                    row.detail,
                    f"{row.max_rel:.10f}",
                    f"{row.p99_rel:.10f}",
                    f"{row.mae:.10f}",
                    f"{row.max_abs:.10f}",
                ]
            )


def aggregate_by_method(rows: list[ResultRow]) -> list[dict[str, object]]:
    aggregates = []
    keys = sorted({(row.method, row.detail) for row in rows})
    for method, detail in keys:
        subset = [row for row in rows if row.method == method and row.detail == detail]
        aggregates.append(
            {
                "method": method,
                "detail": detail,
                "worst_max_rel": max(row.max_rel for row in subset),
                "p95_max_rel": float(np.quantile([row.max_rel for row in subset], 0.95)),
                "avg_p99_rel": float(np.mean([row.p99_rel for row in subset])),
                "avg_mae": float(np.mean([row.mae for row in subset])),
                "count_over_1pct": sum(row.max_rel > 0.01 for row in subset),
                "count_over_3pct": sum(row.max_rel > 0.03 for row in subset),
                "count_over_5pct": sum(row.max_rel > 0.05 for row in subset),
            }
        )
    return sorted(aggregates, key=lambda row: row["worst_max_rel"])


def write_summary(rows: list[ResultRow], elapsed_seconds: float) -> None:
    aggregates = aggregate_by_method(rows)
    worst_rows = sorted(rows, key=lambda row: row.max_rel, reverse=True)[:15]
    n_cases = len(OPTION_TYPES) * len(STRIKES) * len(VOLS) * len(MATURITIES)
    n_state_fits = n_cases * len(TIME_FRACTIONS)

    lines = [
        "# European Option Grid Test",
        "",
        "This sweep tested the no-asymptotic tail-biased MC proxy method across",
        "calls and puts, multiple strikes, volatilities, and maturities.",
        "",
        "Training still used Monte Carlo labels only. Black-Scholes was used only",
        "as the diagnostic benchmark.",
        "",
        "## Setup",
        "",
        f"- option types: `{OPTION_TYPES}`",
        f"- strikes: `{STRIKES}`",
        f"- vols: `{VOLS}`",
        f"- maturities: `{MATURITIES}`",
        f"- time fractions: `{TIME_FRACTIONS}`",
        f"- total option parameter cases: `{n_cases}`",
        f"- total time-slice fits per method: `{n_state_fits}`",
        f"- state points per fit: `{N_STATE_POINTS}`",
        f"- shifted MC paths per state: `{MC_PATHS_PER_STATE:,}`",
        f"- relative error denominator: `max(true_value, {RELATIVE_ERROR_FLOOR:g})`",
        f"- elapsed seconds: `{elapsed_seconds:.1f}`",
        "",
        "## Aggregate Results By Method",
        "",
        "| Method | Detail | Worst Max % Error | P95 Max % Error | Avg P99 % Error | Avg MAE | >1% | >3% | >5% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in aggregates:
        lines.append(
            "| "
            f"{row['method']} | {row['detail']} | "
            f"{100.0 * row['worst_max_rel']:.3f}% | "
            f"{100.0 * row['p95_max_rel']:.3f}% | "
            f"{100.0 * row['avg_p99_rel']:.3f}% | "
            f"{row['avg_mae']:.6f} | "
            f"{row['count_over_1pct']} | "
            f"{row['count_over_3pct']} | "
            f"{row['count_over_5pct']} |"
        )

    lines.extend(
        [
            "",
            "## Worst Individual Rows",
            "",
            "| Option | K | Vol | T | t | Method | Max % Error | P99 % Error | MAE |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )

    for row in worst_rows:
        lines.append(
            "| "
            f"{row.option_type} | {row.strike:.0f} | {row.vol:.2f} | "
            f"{row.maturity:.2f} | {row.time:.2f} | "
            f"{row.method} {row.detail} | "
            f"{100.0 * row.max_rel:.3f}% | "
            f"{100.0 * row.p99_rel:.3f}% | "
            f"{row.mae:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The sweep uses the same shifted-MC state-labeling method as the earlier",
            "  European benchmark.",
            "- No asymptotic proxy, asymptotic anchor, or asymptotic mixing is used.",
            "- The comparison focuses on whether the method remains stable across",
            "  option parameter combinations.",
            "",
        ]
    )

    SUMMARY_PATH.write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rows = []
    combo_index = 0

    for option_type in OPTION_TYPES:
        for strike in STRIKES:
            for vol in VOLS:
                for maturity in MATURITIES:
                    combo_index += 1
                    params = GBMParams(
                        s0=100.0,
                        strike=strike,
                        vol=vol,
                        maturity=maturity,
                        n_steps=5,
                        seed=7,
                        option_type=option_type,
                    )
                    rows.extend(run_case(params, combo_index))
                    print(
                        f"completed {combo_index:02d}: "
                        f"{option_type} K={strike:.0f} vol={vol:.2f} T={maturity:.1f}"
                    )

    elapsed_seconds = perf_counter() - start
    write_csv(rows)
    write_summary(rows, elapsed_seconds)
    aggregates = aggregate_by_method(rows)

    print()
    print("European option grid test complete")
    print(f"rows: {len(rows)}")
    print(f"CSV written to: {CSV_PATH.resolve()}")
    print(f"summary written to: {SUMMARY_PATH.resolve()}")
    print()
    print("Aggregate results:")
    for row in aggregates:
        print(
            f"{row['method']} | {row['detail']}: "
            f"worst max={100.0 * row['worst_max_rel']:.3f}%, "
            f"p95 max={100.0 * row['p95_max_rel']:.3f}%, "
            f">5%={row['count_over_5pct']}"
        )
    print(f"elapsed seconds: {elapsed_seconds:.1f}")


if __name__ == "__main__":
    main()
