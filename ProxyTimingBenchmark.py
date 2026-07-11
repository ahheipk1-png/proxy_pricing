import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from html import escape

import numpy as np
from scipy.interpolate import PchipInterpolator

import AmericanMain
import AsianMain
import AutocallableMain
import BarrierMain
import BasketAsianMain
import BasketCliquetMain
import BermudanMain
import CliquetMain
import EuroMain
import RandomOptionMain
import SLVCliquetMain


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MD = os.path.join(
    ROOT, "Markdown", "MethodStudy", "results", "timing_performance_summary.md"
)
OUTPUT_HTML = os.path.join(
    ROOT, "Markdown", "MethodStudy", "results", "timing_performance_summary.html"
)
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
MIN_EXECUTED_VALUATION_CASES = 100
MIN_TIMED_SCENARIO_COMBINATIONS = 100

REFERENCE = {
    "European": {
        "method": "log Chebyshev in d1, degree 7",
        "doc_cases": "216",
        "coverage": "54 parameter cases x 4 time slices",
        "worst": "1.029%",
        "p99": "0.256%",
        "source": "../../European/findings/european_option_grid/summary.md",
        "subtests": [
            "2 option types x 3 strikes x 3 volatilities x 3 maturities x 4 time slices = 216 combinations",
        ],
        "cases": [
            "Calls and puts",
            "Strikes: 80, 100, 120",
            "Vols: 10%, 20%, 40%",
            "Maturities: 0.5, 1.0, 2.0 years",
            "Time fractions: 0.2, 0.4, 0.6, 0.8",
        ],
    },
    "Asian": {
        "method": "adjusted-moneyness PCHIP hybrid",
        "doc_cases": "5",
        "coverage": "5 fixing dates",
        "worst": "2.454%",
        "p99": "1.141%",
        "source": "../../Asian/results/summary.md",
        "subtests": [
            "1 monthly arithmetic Asian setup x 5 fixing dates = 5 combinations",
        ],
        "cases": ["Monthly arithmetic Asian call", "Day indices: 0, 3, 6, 9, 11"],
    },
    "American": {
        "method": "PCHIP continuation spline",
        "doc_cases": "6",
        "coverage": "6 exercise-time slices",
        "worst": "2.249%",
        "p99": "0.947%",
        "source": "../../American/results/summary.md",
        "subtests": [
            "1 American put setup x 6 exercise-time slices = 6 combinations",
        ],
        "cases": ["American put", "Exercise steps tested: 0, 20, 40, 60, 80, 100"],
    },
    "Bermudan": {
        "method": "log-continuation PCHIP dynamic programming",
        "doc_cases": "25",
        "coverage": "5 parameter cases x 5 exercise dates",
        "worst": "8.300%",
        "p99": "1.924%",
        "source": "../../Bermudan/results/summary.md",
        "subtests": [
            "5 parameter cases x 5 exercise dates = 25 combinations",
        ],
        "cases": [
            "base_put, low_vol, high_vol, deep_itm, dividend_rich",
            "Exercise indices: 0, 3, 6, 9, 12",
        ],
    },
    "Barrier": {
        "method": "best of PCHIP/Akima/Bernstein by variant",
        "doc_cases": "30",
        "coverage": "3 barrier types x 2 monitoring types x 5 months",
        "worst": "8.133%-9.096%",
        "p99": "timing rerun",
        "source": "../../Barrier/results/summary.md",
        "subtests": [
            "3 barrier types x 2 monitoring styles x 5 reset months = 30 combinations",
        ],
        "cases": [
            "Barrier types: down-out, up-out, double-out",
            "Monitoring: discrete and continuous",
            "Months: 0, 3, 6, 9, 12",
        ],
    },
    "Cliquet": {
        "method": "bounded-logit Chebyshev on expected-total z",
        "doc_cases": "5",
        "coverage": "5 reset dates",
        "worst": "3.593%",
        "p99": "0.928%",
        "source": "../../Cliquet/results/summary.md",
        "subtests": [
            "1 GBM cliquet setup x 5 reset dates = 5 combinations",
        ],
        "cases": ["Monthly GBM cliquet", "Day indices: 0, 3, 6, 9, 12"],
    },
    "SLV cliquet": {
        "method": "adaptive hybrid",
        "doc_cases": "5",
        "coverage": "5 reset dates",
        "worst": "5.067%",
        "p99": "2.760%",
        "source": "../../SLVCliquet/results/summary.md",
        "subtests": [
            "1 SLV cliquet setup x 5 reset months = 5 combinations",
        ],
        "cases": ["Single-underlying SLV cliquet", "Reset months: 0, 3, 6, 9, 12"],
    },
    "Basket Asian": {
        "method": "PCHIP-calibrated PCA log-factor correction",
        "doc_cases": "6",
        "coverage": "6 fixing dates",
        "worst": "5.873%",
        "p99": "1.820%",
        "source": "../../BasketAsian/results/summary.md",
        "subtests": [
            "1 ten-underlying basket setup x 6 fixing dates = 6 combinations",
        ],
        "cases": [
            "10-underlying arithmetic basket Asian",
            "Mixed positive and negative correlations",
            "Day indices: 0, 3, 6, 9, 10, 11",
        ],
    },
    "Basket cliquet": {
        "method": "cached Sobol/LR safety proxy for hard cases",
        "doc_cases": "55",
        "coverage": "40 fitted variant/month checks plus 15 hard-case safety checks",
        "worst": "11.2%",
        "p99": "timing rerun",
        "source": "../../BasketCliquet/results/summary.md",
        "subtests": [
            "8 generalized coupon variants x 5 reset months = 40 fitted-proxy combinations",
            "5 hard safety-proxy variants x 3 reset months = 15 safety combinations",
            "40 + 15 = 55 documented combinations",
        ],
        "cases": [
            "Fitted variants: basket_return, weighted_average, basket_ratio, average_clipped, second_worst, worst_of, best_of, spread_bonus",
            "Safety checks: 5 hard variants x reset months 3, 6, 9",
        ],
    },
    "Autocallable": {
        "method": "Akima/PCHIP log-value interpolation",
        "doc_cases": "20",
        "coverage": "5 product cases x 4 observation dates",
        "worst": "0.123%",
        "p99": "0.051%",
        "source": "../../Autocallable/results/summary.md",
        "subtests": [
            "5 autocallable product cases x 4 observation dates = 20 combinations",
        ],
        "cases": [
            "base, high_vol, low_autocall, high_coupon, downside_heavy",
            "Observation indices: 0, 1, 2, 3",
        ],
    },
    "Random payoff": {
        "method": "Akima log-value interpolation",
        "doc_cases": "96",
        "coverage": "8 random payoffs x 3 markets x 4 times",
        "worst": "0.658%",
        "p99": "0.037%",
        "source": "../../RandomOption/results/summary.md",
        "subtests": [
            "8 random payoff shapes x 3 market regimes x 4 time fractions = 96 combinations",
        ],
        "cases": [
            "8 fixed-seed piecewise-linear terminal payoffs",
            "Markets: base, low-vol, high-vol",
            "Time fractions: 0.0, 0.25, 0.5, 0.75",
        ],
    },
}


def elapsed(callable_obj, *args, **kwargs):
    start = time.perf_counter()
    value = callable_obj(*args, **kwargs)
    return value, time.perf_counter() - start


def score(prediction, truth, floor):
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    error = prediction - truth
    abs_error = np.abs(error)
    rel_error = abs_error / np.maximum(np.abs(truth), floor)
    return {
        "max_rel": float(np.max(rel_error)),
        "p99_rel": float(np.quantile(rel_error, 0.99)),
        "mae": float(np.mean(abs_error)),
    }


def add_result(rows, option, method, case, sample_sec, train_sec, prediction, truth, floor):
    truth_array = np.asarray(truth, dtype=float)
    metrics = score(prediction, truth, floor)
    rows.append(
        {
            "option": option,
            "method": method,
            "case": case,
            "case_count": int(truth_array.size),
            "sample_sec": float(sample_sec),
            "train_sec": float(train_sec),
            **metrics,
        }
    )


def benchmark_european(rows):
    base = EuroMain.Params()
    old_paths = EuroMain.MC_PATHS_PER_STATE
    EuroMain.MC_PATHS_PER_STATE = 2048
    try:
        case_index = 0
        for option_type in ("call", "put"):
            for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
                for vol in (0.12, 0.18, 0.24, 0.32, 0.45):
                    for maturity in (0.5, 1.0):
                        params = replace(
                            base,
                            option_type=option_type,
                            strike=strike,
                            vol=vol,
                            maturity=maturity,
                            seed=base.seed + case_index,
                        )
                        rng = np.random.default_rng(params.seed + 900)
                        for fraction in (0.25, 0.65):
                            tau = params.maturity * (1.0 - fraction)
                            train_spot = EuroMain.delta_space_spot_grid(tau, params, 41)
                            train_values, sample_sec = elapsed(
                                EuroMain.shifted_mc_value, train_spot, tau, params, rng
                            )
                            proxy, train_sec = elapsed(
                                EuroMain.fit_log_pchip_d1,
                                train_spot,
                                train_values,
                                tau,
                                params,
                            )
                            test_spot = EuroMain.delta_space_spot_grid(tau, params, 31)
                            truth = EuroMain.black_scholes_value(test_spot, tau, params)
                            add_result(
                                rows,
                                "European",
                                "log PCHIP in d1",
                                (
                                    f"{option_type}_K={strike:.0f}_vol={vol:.2f}_"
                                    f"T={maturity:.1f}_t={fraction:.2f}"
                                ),
                                sample_sec,
                                train_sec,
                                proxy(test_spot),
                                truth,
                                EuroMain.RELATIVE_ERROR_FLOOR,
                            )
                        case_index += 1
    finally:
        EuroMain.MC_PATHS_PER_STATE = old_paths


def benchmark_asian(rows):
    base = AsianMain.Params()
    case_index = 0
    for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
        for vol in (0.12, 0.18, 0.24, 0.32):
            for day in (0, 3, 6, 9, 11):
                params = replace(
                    base,
                    option_type="call",
                    strike=strike,
                    vol=vol,
                    seed=base.seed + case_index,
                )
                rng = np.random.default_rng(params.seed + 901)
                train_spot, train_sum = AsianMain.make_adjusted_moneyness_grid(
                    day, params, 31
                )
                train_pack, sample_sec = elapsed(
                    AsianMain.build_labels,
                    train_spot,
                    train_sum,
                    day,
                    params,
                    rng,
                    1024,
                )
                train_values, _ = train_pack
                proxy, train_sec = elapsed(
                    AsianMain.fit_adjusted_hybrid_proxy,
                    train_spot,
                    train_sum,
                    train_values,
                    day,
                    params,
                )
                test_spot, test_sum = AsianMain.make_state_grid(day, params, 5, 3)
                benchmark, _ = AsianMain.build_labels(
                    test_spot,
                    test_sum,
                    day,
                    params,
                    rng,
                    4096,
                )
                add_result(
                    rows,
                    "Asian",
                    "adjusted-moneyness PCHIP hybrid",
                    f"call_K={strike:.0f}_vol={vol:.2f}_day={day}",
                    sample_sec,
                    train_sec,
                    proxy(test_spot, test_sum),
                    benchmark,
                    AsianMain.RELATIVE_ERROR_FLOOR,
                )
                case_index += 1


def train_american_timed(params, steps=24, states=61, paths=512):
    old = {
        "EXERCISE_STEPS": AmericanMain.EXERCISE_STEPS,
        "TRAIN_STATES": AmericanMain.TRAIN_STATES,
        "PATHS_PER_STATE": AmericanMain.PATHS_PER_STATE,
        "SPOT_MIN": AmericanMain.SPOT_MIN,
        "SPOT_MAX": AmericanMain.SPOT_MAX,
    }
    AmericanMain.EXERCISE_STEPS = steps
    AmericanMain.TRAIN_STATES = states
    AmericanMain.PATHS_PER_STATE = paths
    AmericanMain.SPOT_MIN = 40.0
    AmericanMain.SPOT_MAX = 220.0
    try:
        normals = AmericanMain.sobol_normals(paths, steps, params.seed).T
        dt = params.maturity / steps
        df = math.exp(-params.rate * dt)
        drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
        vol_step = params.vol * math.sqrt(dt)
        spots = AmericanMain.training_spots()
        value = lambda new_spot: AmericanMain.intrinsic(new_spot, params)
        sample_sec = 0.0
        train_sec = 0.0
        for step in range(steps - 1, -1, -1):
            z = normals[step]
            start = time.perf_counter()
            up = spots[:, None] * np.exp(drift + vol_step * z)[None, :]
            down = spots[:, None] * np.exp(drift - vol_step * z)[None, :]
            continuation = 0.5 * df * np.mean(value(up) + value(down), axis=1)
            sample_sec += time.perf_counter() - start

            start = time.perf_counter()
            log_spots = np.log(spots)
            curve = PchipInterpolator(log_spots, continuation, extrapolate=True)
            log_min, log_max = float(log_spots[0]), float(log_spots[-1])

            def current_value(new_spot, curve=curve, log_min=log_min, log_max=log_max):
                safe_spot = np.maximum(new_spot, 1e-12)
                x = np.clip(np.log(safe_spot), log_min, log_max)
                return np.maximum(
                    AmericanMain.intrinsic(safe_spot, params),
                    np.maximum(curve(x), 0.0),
                )

            value = current_value
            train_sec += time.perf_counter() - start
        return value, sample_sec, train_sec
    finally:
        for name, value in old.items():
            setattr(AmericanMain, name, value)


def american_tree_put(spot, params, steps=240):
    dt = params.maturity / steps
    u = math.exp(params.vol * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((params.rate - params.div_yield) * dt)
    p = min(max((growth - d) / (u - d), 0.0), 1.0)
    df = math.exp(-params.rate * dt)
    j = np.arange(steps + 1)
    nodes = spot * (u**j) * (d ** (steps - j))
    values = AmericanMain.intrinsic(nodes, params)
    for step in range(steps - 1, -1, -1):
        values = df * (p * values[1:] + (1.0 - p) * values[:-1])
        j = np.arange(step + 1)
        nodes = spot * (u**j) * (d ** (step - j))
        values = np.maximum(values, AmericanMain.intrinsic(nodes, params))
    return float(values[0])


def benchmark_american(rows):
    base = AmericanMain.Params()
    case_index = 0
    spots = np.exp(np.linspace(np.log(50.0), np.log(190.0), 21))
    for vol in (0.12, 0.18, 0.24, 0.32, 0.45):
        for rate in (0.01, 0.03, 0.05, 0.08):
            for div_yield in (0.00, 0.01, 0.02, 0.04, 0.06):
                params = replace(
                    base,
                    vol=vol,
                    rate=rate,
                    div_yield=div_yield,
                    seed=base.seed + case_index,
                )
                proxy, sample_sec, train_sec = train_american_timed(
                    params, steps=16, states=41, paths=256
                )
                truth = np.array(
                    [american_tree_put(float(spot), params, steps=120) for spot in spots]
                )
                add_result(
                    rows,
                    "American",
                    "PCHIP dynamic programming",
                    f"put_vol={vol:.2f}_r={rate:.2f}_q={div_yield:.2f}",
                    sample_sec,
                    train_sec,
                    proxy(spots),
                    truth,
                    AmericanMain.RELATIVE_ERROR_FLOOR,
                )
                case_index += 1


def train_bermudan_timed(params, states=121, paths=4096):
    original = {
        "TRAIN_STATES": BermudanMain.TRAIN_STATES,
        "PATHS_PER_STATE": BermudanMain.PATHS_PER_STATE,
    }
    BermudanMain.TRAIN_STATES = states
    BermudanMain.PATHS_PER_STATE = paths
    try:
        dt = params.maturity / params.n_exercise_dates
        df = math.exp(-params.rate * dt)
        drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
        vol_step = params.vol * math.sqrt(dt)
        normals = BermudanMain.sobol_normals(paths, params.n_exercise_dates, params.seed).T
        spots = BermudanMain.training_spots()
        value = lambda new_spot: BermudanMain.intrinsic(new_spot, params)
        sample_sec = 0.0
        train_sec = 0.0
        for step in range(params.n_exercise_dates - 1, -1, -1):
            z = normals[step]
            start = time.perf_counter()
            threshold = (np.log(params.strike / spots) - drift) / vol_step
            shift = np.clip(
                threshold - BermudanMain.IMPORTANCE_SHIFT_BUFFER,
                -BermudanMain.IMPORTANCE_SHIFT_CAP,
                0.0,
            )
            half = max(len(z) // 2, 1)
            base = z[:half]
            actual = np.concatenate(
                (base[None, :] + 0.0 * shift[:, None], base[None, :] + shift[:, None]),
                axis=1,
            )
            shifted_over_base = np.exp(
                np.clip(actual * shift[:, None] - 0.5 * shift[:, None] ** 2, -50.0, 50.0)
            )
            likelihood_ratio = 1.0 / (0.5 + 0.5 * shifted_over_base)
            next_spot = spots[:, None] * np.exp(drift + vol_step * actual)
            continuation = df * np.mean(value(next_spot) * likelihood_ratio, axis=1)
            sample_sec += time.perf_counter() - start

            start = time.perf_counter()
            log_spots = np.log(spots)
            target = np.log(np.maximum(continuation, 0.0) + 1e-12)
            curve = PchipInterpolator(log_spots, target, extrapolate=True)
            log_min, log_max = float(log_spots[0]), float(log_spots[-1])

            def current_value(new_spot, curve=curve, log_min=log_min, log_max=log_max):
                safe_spot = np.maximum(new_spot, 1e-12)
                x = np.clip(np.log(safe_spot), log_min, log_max)
                continuation_value = np.maximum(
                    np.exp(np.clip(curve(x), -45.0, 20.0)) - 1e-12, 0.0
                )
                return np.maximum(BermudanMain.intrinsic(safe_spot, params), continuation_value)

            value = current_value
            train_sec += time.perf_counter() - start
        return value, sample_sec, train_sec
    finally:
        for name, value in original.items():
            setattr(BermudanMain, name, value)


def benchmark_bermudan(rows):
    base = BermudanMain.Params()
    original_steps = BermudanMain.BINOMIAL_STEPS_PER_PERIOD
    BermudanMain.BINOMIAL_STEPS_PER_PERIOD = 8
    spots = np.exp(np.linspace(np.log(50.0), np.log(190.0), 21))
    case_index = 0
    try:
        for vol in (0.14, 0.18, 0.24, 0.32, 0.42):
            for rate in (0.01, 0.03, 0.05, 0.08):
                for div_yield in (0.00, 0.01, 0.02, 0.04, 0.06):
                    params = replace(
                        base,
                        vol=vol,
                        rate=rate,
                        div_yield=div_yield,
                        seed=base.seed + case_index,
                    )
                    proxy, sample_sec, train_sec = train_bermudan_timed(
                        params, states=51, paths=512
                    )
                    truth = BermudanMain.benchmark_values(spots, 0, params)
                    add_result(
                        rows,
                        "Bermudan",
                        "log-continuation PCHIP DP",
                        f"put_vol={vol:.2f}_r={rate:.2f}_q={div_yield:.2f}",
                        sample_sec,
                        train_sec,
                        proxy(spots),
                        truth,
                        BermudanMain.RELATIVE_ERROR_FLOOR,
                    )
                    case_index += 1
    finally:
        BermudanMain.BINOMIAL_STEPS_PER_PERIOD = original_steps


def benchmark_barrier(rows):
    base = BarrierMain.Params()
    best_method = {
        ("down_out", "discrete"): "bernstein",
        ("down_out", "continuous"): "akima",
        ("up_out", "discrete"): "pchip",
        ("up_out", "continuous"): "pchip",
        ("double_out", "discrete"): "bernstein",
        ("double_out", "continuous"): "akima",
    }
    case_index = 0
    for vol in (0.14, 0.18, 0.24, 0.32):
        for month in (0, 3, 6, 9, 12):
            for kind in ("down_out", "up_out", "double_out"):
                params = replace(base, vol=vol, seed=base.seed + case_index)
                rng = np.random.default_rng(params.seed + 902)
                train_spots = BarrierMain.make_spot_grid(kind, params, 41)
                train_pack, sample_sec = elapsed(
                    BarrierMain.build_labels,
                    train_spots,
                    month,
                    kind,
                    params,
                    rng,
                    1024,
                )
                train_values, _ = train_pack
                test_spots = BarrierMain.make_shifted_spot_grid(kind, params, 15)
                benchmark, _ = BarrierMain.build_labels(
                    test_spots, month, kind, params, rng, 4096
                )
                for monitoring in BarrierMain.MONITORING_TYPES:
                    method = best_method[(kind, monitoring)]
                    proxy, train_sec = elapsed(
                        BarrierMain.fit_proxy,
                        train_spots,
                        train_values[monitoring],
                        kind,
                        params,
                        method,
                    )
                    add_result(
                        rows,
                        "Barrier",
                        "best mapped knock-out interpolator",
                        f"{kind}_{monitoring}_vol={vol:.2f}_month={month}",
                        sample_sec / len(BarrierMain.MONITORING_TYPES),
                        train_sec,
                        proxy(test_spots),
                        benchmark[monitoring],
                        BarrierMain.RELATIVE_ERROR_FLOOR,
                    )
                case_index += 1


def benchmark_cliquet(rows):
    base = CliquetMain.Params()
    case_index = 0
    for vol in (0.14, 0.18, 0.24, 0.32, 0.42):
        for rate in (0.01, 0.03, 0.05, 0.08):
            for day in (0, 3, 6, 9, 12):
                params = replace(
                    base,
                    vol=vol,
                    rate=rate,
                    seed=base.seed + case_index,
                )
                rng = np.random.default_rng(params.seed + 903)
                train_grid = CliquetMain.make_boundary_accrued_grid(day, params, 41)
                train_pack, sample_sec = elapsed(
                    CliquetMain.build_labels, train_grid, day, params, rng, 1024
                )
                train_values, _ = train_pack
                proxy, train_sec = elapsed(
                    CliquetMain.fit_default_proxy, train_grid, train_values, day, params
                )
                test_grid = CliquetMain.make_boundary_accrued_grid(day, params, 15)
                benchmark, _ = CliquetMain.build_labels(
                    test_grid, day, params, rng, 4096
                )
                add_result(
                    rows,
                    "Cliquet",
                    "bounded-logit Chebyshev",
                    f"vol={vol:.2f}_r={rate:.2f}_day={day}",
                    sample_sec,
                    train_sec,
                    proxy(test_grid),
                    benchmark,
                    CliquetMain.RELATIVE_ERROR_FLOOR,
                )
                case_index += 1


def benchmark_slv_cliquet(rows):
    base = SLVCliquetMain.Params()
    case_index = 0
    for v0 in (0.02, 0.0324, 0.045, 0.060, 0.080):
        for vol_of_var in (0.20, 0.30, 0.45, 0.60):
            for month in (0, 3, 6, 9, 12):
                params = replace(
                    base,
                    v0=v0,
                    theta=v0,
                    vol_of_var=vol_of_var,
                    seed=base.seed + case_index,
                )
                rng = np.random.default_rng(params.seed + 904)
                train_states = SLVCliquetMain.make_states(month, params, 41)
                train_pack, sample_sec = elapsed(
                    SLVCliquetMain.labels, train_states, month, params, rng, 256
                )
                train_values, _ = train_pack
                proxy, train_sec = elapsed(
                    SLVCliquetMain.fit_default, train_states, train_values, month, params
                )
                test_states = SLVCliquetMain.make_states(
                    month, params, 15, validation=True
                )
                benchmark, _ = SLVCliquetMain.labels(
                    test_states, month, params, rng, 1024
                )
                add_result(
                    rows,
                    "SLV cliquet",
                    "adaptive hybrid proxy",
                    f"v0={v0:.4f}_vov={vol_of_var:.2f}_month={month}",
                    sample_sec,
                    train_sec,
                    proxy(test_states),
                    benchmark,
                    SLVCliquetMain.RELATIVE_ERROR_FLOOR,
                )
                case_index += 1


def benchmark_basket_asian(rows):
    base = BasketAsianMain.Params()
    case_index = 0
    for strike in (85.0, 95.0, 100.0, 110.0, 125.0):
        for rate in (0.01, 0.03, 0.05, 0.08):
            for day in (0, 3, 6, 9, 11):
                params = replace(
                    base,
                    strike=strike,
                    rate=rate,
                    seed=base.seed + case_index,
                )
                train_spots, train_running = BasketAsianMain.make_states(day, params, 31)
                train_pack, sample_sec = elapsed(
                    BasketAsianMain.simulate_labels,
                    train_spots,
                    train_running,
                    day,
                    params,
                    512,
                    params.seed + 20_000 + day,
                    True,
                )
                train_values, _ = train_pack
                proxy, train_sec = elapsed(
                    BasketAsianMain.fit_proxy,
                    train_spots,
                    train_running,
                    train_values,
                    day,
                    params,
                    "pchip_calibrated_log_factor_pca",
                )
                test_spots, test_running = BasketAsianMain.make_states(
                    day, params, 9, validation=True
                )
                benchmark, _ = BasketAsianMain.simulate_labels(
                    test_spots,
                    test_running,
                    day,
                    params,
                    2048,
                    params.seed + 30_000 + day,
                    True,
                )
                add_result(
                    rows,
                    "Basket Asian",
                    "PCHIP-calibrated PCA correction",
                    f"K={strike:.0f}_r={rate:.2f}_day={day}",
                    sample_sec,
                    train_sec,
                    proxy(test_spots, test_running),
                    benchmark,
                    BasketAsianMain.RELATIVE_ERROR_FLOOR,
                )
                case_index += 1


def benchmark_basket_cliquet(rows):
    base = BasketCliquetMain.Params()
    variants = ("basket_return", "basket_ratio", "second_worst", "worst_of", "best_of")
    cap_floor_pairs = ((-0.02, 0.04), (-0.03, 0.05), (-0.01, 0.03), (-0.04, 0.06))
    case_index = 0
    for local_floor, local_cap in cap_floor_pairs:
        for month in (0, 3, 6, 9, 12):
            params = replace(
                base,
                local_floor=local_floor,
                local_cap=local_cap,
                seed=base.seed + case_index,
            )
            rng = np.random.default_rng(params.seed + 905)
            spot_z = BasketCliquetMain.feature_normals(params, 512)
            train_states, train_moments = BasketCliquetMain.make_states(
                month, params, 45, spot_z
            )
            train_pack, sample_sec = elapsed(
                BasketCliquetMain.build_labels,
                train_states,
                train_moments,
                month,
                params,
                rng,
                192,
            )
            train_values, _ = train_pack
            test_states, test_moments = BasketCliquetMain.make_states(
                month, params, 11, spot_z, validation=True
            )
            benchmark, _ = BasketCliquetMain.build_labels(
                test_states, test_moments, month, params, rng, 512
            )
            for variant in variants:
                proxy, train_sec = elapsed(
                    BasketCliquetMain.fit_proxy,
                    train_states,
                    train_moments,
                    train_values[variant],
                    month,
                    variant,
                    params,
                    "adaptive_blend",
                )
                add_result(
                    rows,
                    "Basket cliquet",
                    "adaptive blend fitted proxy",
                    (
                        f"{variant}_floor={local_floor:.2f}_"
                        f"cap={local_cap:.2f}_month={month}"
                    ),
                    sample_sec / len(variants),
                    train_sec,
                    proxy(test_states, test_moments),
                    benchmark[variant],
                    BasketCliquetMain.RELATIVE_ERROR_FLOOR,
                )
            case_index += 1


def benchmark_autocallable(rows):
    base = AutocallableMain.Params()
    train_grid = AutocallableMain.training_spots()
    test_grid = AutocallableMain.validation_spots()
    case_index = 0
    for vol in (0.16, 0.22, 0.28, 0.34, 0.42):
        for protection in (0.55, 0.60, 0.65, 0.70, 0.75):
            params = replace(
                base,
                vol=vol,
                protection_barrier=protection,
                seed=base.seed + case_index,
            )
            for obs_index in AutocallableMain.TEST_OBSERVATION_INDICES:
                train_values, sample_sec = elapsed(
                    AutocallableMain.simulate_values,
                    train_grid,
                    obs_index,
                    params,
                    1024,
                    params.seed + 17 * obs_index,
                )
                proxy, train_sec = elapsed(
                    AutocallableMain.fit_curve, "akima", train_grid, train_values
                )
                truth = AutocallableMain.simulate_values(
                    test_grid,
                    obs_index,
                    params,
                    4096,
                    params.seed + 1000 + 37 * obs_index,
                )
                add_result(
                    rows,
                    "Autocallable",
                    "Akima log-value interpolation",
                    f"vol={vol:.2f}_prot={protection:.2f}_obs={obs_index}",
                    sample_sec,
                    train_sec,
                    proxy(test_grid),
                    truth,
                    AutocallableMain.RELATIVE_ERROR_FLOOR,
                )
            case_index += 1


def benchmark_random_option(rows):
    markets = list(RandomOptionMain.MARKET_CASES) + [
        ("mid_high_vol", RandomOptionMain.Market(vol=0.30, seed=425)),
    ]
    for market_name, market in markets:
        for payoff_case in RandomOptionMain.PAYOFF_CASES:
            train_spots = RandomOptionMain.spot_grid(RandomOptionMain.TRAIN_STATES)
            test_spots = RandomOptionMain.spot_grid(RandomOptionMain.VALIDATION_STATES)
            for time_fraction in RandomOptionMain.TIME_FRACTIONS:
                tau = market.maturity * (1.0 - time_fraction)
                train_values, sample_sec = elapsed(
                    RandomOptionMain.mc_value,
                    train_spots,
                    tau,
                    market,
                    payoff_case,
                    2048,
                    market.seed + int(1000 * time_fraction),
                )
                proxy, train_sec = elapsed(
                    RandomOptionMain.fit_proxy, "akima", train_spots, train_values
                )
                truth = RandomOptionMain.mc_value(
                    test_spots,
                    tau,
                    market,
                    payoff_case,
                    8192,
                    market.seed + 10_000 + int(1000 * time_fraction),
                )
                add_result(
                    rows,
                    "Random payoff",
                    "Akima log-value interpolation",
                    f"{market_name}_{payoff_case.name}_t={time_fraction:.2f}",
                    sample_sec,
                    train_sec,
                    proxy(test_spots),
                    truth,
                    RandomOptionMain.RELATIVE_ERROR_FLOOR,
                )


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["option"]].append(row)
    output = []
    for option, subset in grouped.items():
        methods = sorted({row["method"] for row in subset})
        output.append(
            {
                "option": option,
                "method": methods[0] if len(methods) == 1 else ", ".join(methods),
                "fits": len(subset),
                "valuation_cases": int(np.sum([row["case_count"] for row in subset])),
                "avg_max_rel": float(np.mean([row["max_rel"] for row in subset])),
                "worst_max_rel": float(np.max([row["max_rel"] for row in subset])),
                "avg_p99_rel": float(np.mean([row["p99_rel"] for row in subset])),
                "avg_mae": float(np.mean([row["mae"] for row in subset])),
                "avg_sample_sec": float(np.mean([row["sample_sec"] for row in subset])),
                "avg_train_sec": float(np.mean([row["train_sec"] for row in subset])),
                "timed_rows": subset,
            }
        )
    order = [
        "European",
        "Asian",
        "American",
        "Bermudan",
        "Barrier",
        "Cliquet",
        "SLV cliquet",
        "Basket Asian",
        "Basket cliquet",
        "Autocallable",
        "Random payoff",
    ]
    rank = {name: index for index, name in enumerate(order)}
    return sorted(output, key=lambda row: rank.get(row["option"], 999))


def benchmark_registry():
    return (
        ("European", benchmark_european),
        ("Asian", benchmark_asian),
        ("American", benchmark_american),
        ("Bermudan", benchmark_bermudan),
        ("Barrier", benchmark_barrier),
        ("Cliquet", benchmark_cliquet),
        ("SLV cliquet", benchmark_slv_cliquet),
        ("Basket Asian", benchmark_basket_asian),
        ("Basket cliquet", benchmark_basket_cliquet),
        ("Autocallable", benchmark_autocallable),
        ("Random payoff", benchmark_random_option),
    )


def run_benchmark_worker(option_name):
    benchmarks = dict(benchmark_registry())
    rows = []
    start = time.perf_counter()
    benchmarks[option_name](rows)
    return option_name, rows, time.perf_counter() - start


def run_benchmarks(worker_count):
    rows = []
    timings = {}
    if worker_count == 1:
        for name, benchmark in benchmark_registry():
            start = time.perf_counter()
            benchmark(rows)
            timings[name] = time.perf_counter() - start
        return rows, timings

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(run_benchmark_worker, name): name
            for name, _ in benchmark_registry()
        }
        for future in as_completed(futures):
            name, worker_rows, elapsed_seconds = future.result()
            rows.extend(worker_rows)
            timings[name] = elapsed_seconds
    return rows, timings


def markdown_table(summary, total_seconds, worker_count, worker_timings):
    lines = [
        "# Timing And Average Performance Summary",
        "",
        "This table is generated by `ProxyTimingBenchmark.py`. It uses representative",
        "reduced-budget runs with the real pricing/proxy code so every option family",
        "can be timed in one pass. It is not the 10M/20M-path production grid.",
        "",
        "The documented performance columns come from the current product-specific",
        "summaries. The smoke accuracy columns are only a sanity check for the",
        "reduced-budget timing pass; they can be noisy for tail/barrier/high-dimensional",
        "products.",
        "",
        "Timing columns separate training-label/sample generation from proxy fitting.",
        "Benchmark generation time is excluded from both timing columns.",
        "",
        "For products whose original markdown did not store an aggregate p99,",
        "this script fills the p99 cell from the current timing rerun and marks",
        "it with `(timing rerun)`.",
        "",
        "HTML dashboard: `timing_performance_summary.html`.",
        "",
        f"Total benchmark wall time: `{total_seconds:.2f}` seconds.",
        f"Worker processes: `{worker_count}`.",
        f"Minimum timed scenario combinations per option family: `{MIN_TIMED_SCENARIO_COMBINATIONS}`.",
        "",
        "| Option type | Timed scenario combinations | Executed valuation states | Test-case details | Documented default/best method | Documented worst max | Documented avg p99 | Smoke avg max | Smoke worst max | Smoke avg p99 | Avg sample gen / fit | Avg proxy train / fit |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        reference = REFERENCE.get(
            row["option"],
            {
                "method": row["method"],
                "doc_cases": "n/a",
                "coverage": "n/a",
                "worst": "n/a",
                "p99": "n/a",
                "source": "",
                "cases": [],
            },
        )
        anchor = anchor_id(row["option"])
        source = reference["source"]
        detail_link = f"[view cases](timing_performance_summary.html#{anchor})"
        method = reference["method"]
        if source:
            method = f"[{method}]({source})"
        reference_p99 = reference_p99_display(reference, row)
        lines.append(
            f"| {row['option']} | {row['fits']} | {row['valuation_cases']} | {detail_link} | "
            f"{method} | {reference['worst']} | {reference_p99} | "
            f"{100.0 * row['avg_max_rel']:.3f}% | "
            f"{100.0 * row['worst_max_rel']:.3f}% | "
            f"{100.0 * row['avg_p99_rel']:.3f}% | "
            f"{row['avg_sample_sec']:.4f}s | "
            f"{row['avg_train_sec']:.4f}s |"
        )
    lines.extend(
        [
            "",
            "## Source Markdown Summary Breakdown",
            "",
            "| Option type | Source-summary combination breakdown | Original source-summary combinations |",
            "|---|---|---:|",
        ]
    )
    for row in summary:
        reference = REFERENCE[row["option"]]
        breakdown = "<br>".join(reference["subtests"])
        lines.append(
            f"| {row['option']} | {breakdown} | {reference['doc_cases']} |"
        )
    lines.extend(
        [
            "",
            "## Expanded Timed Scenario Breakdown",
            "",
            "| Option type | Timed scenario combinations | Executed valuation states | Average valuation states per combination |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary:
        avg_cases = row["valuation_cases"] / max(row["fits"], 1)
        lines.append(
            f"| {row['option']} | {row['fits']} | {row['valuation_cases']} | {avg_cases:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Worker Timing Breakdown",
            "",
            "| Option type | Worker wall time |",
            "|---|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['option']} | {worker_timings.get(row['option'], 0.0):.2f}s |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Avg sample gen / fit` is the time to create training labels or continuation",
            "  samples for one fitted proxy/time slice.",
            "- `Avg proxy train / fit` is the time spent fitting/interpolating/regressing",
            "  after labels are available.",
            "- A timed scenario combination is one option parameter/date/variant row",
            "  that fits a proxy and validates it on a small state grid.",
            "- Shared label-generation runs, such as barrier discrete/continuous labels",
            "  and basket-cliquet variant labels, are amortized across the fitted rows",
            "  that reuse the same samples.",
            "- For dynamic-programming products, one combination means the whole trained proxy at",
            "  the tested valuation date, including all backward induction steps.",
            "- Barrier and basket cliquet did not have an aggregate p99 in their original",
            "  source summaries, so their p99 cells are filled from this timing rerun.",
        ]
    )
    return "\n".join(lines) + "\n"


def reference_p99_display(reference, row):
    if reference["p99"] == "timing rerun":
        return f"{100.0 * row['avg_p99_rel']:.3f}% (timing rerun)"
    return reference["p99"]


def anchor_id(option):
    return (
        option.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("_", "-")
    ) + "-cases"


def html_table(summary, total_seconds, worker_count, worker_timings):
    rows_html = []
    for row in summary:
        reference = REFERENCE[row["option"]]
        reference_p99 = reference_p99_display(reference, row)
        rows_html.append(
            "<tr>"
            f"<td>{escape(row['option'])}</td>"
            f"<td class='num'>{row['fits']}</td>"
            f"<td class='num'>{row['valuation_cases']}</td>"
            f"<td><a href='#{anchor_id(row['option'])}'>view cases</a></td>"
            f"<td><a href='{escape(reference['source'])}'>{escape(reference['method'])}</a></td>"
            f"<td class='num'>{escape(reference['worst'])}</td>"
            f"<td class='num'>{escape(reference_p99)}</td>"
            f"<td class='num'>{100.0 * row['avg_max_rel']:.3f}%</td>"
            f"<td class='num'>{100.0 * row['worst_max_rel']:.3f}%</td>"
            f"<td class='num'>{100.0 * row['avg_p99_rel']:.3f}%</td>"
            f"<td class='num'>{row['avg_sample_sec']:.4f}s</td>"
            f"<td class='num'>{row['avg_train_sec']:.4f}s</td>"
            "</tr>"
        )

    detail_sections = []
    for row in summary:
        reference = REFERENCE[row["option"]]
        case_items = "\n".join(
            f"<li>{escape(item)}</li>" for item in reference["cases"]
        )
        subtest_items = "\n".join(
            f"<li>{escape(item)}</li>" for item in reference["subtests"]
        )
        matching_rows = [
            timed for timed in row.get("timed_rows", []) if timed["option"] == row["option"]
        ]
        timed_items = "\n".join(
            f"<li><code>{escape(timed['case'])}</code>: "
            f"{timed['case_count']} valuation cases, "
            f"max {100.0 * timed['max_rel']:.3f}%, "
            f"p99 {100.0 * timed['p99_rel']:.3f}%, "
            f"sample {timed['sample_sec']:.4f}s, train {timed['train_sec']:.4f}s</li>"
            for timed in matching_rows
        )
        detail_sections.append(
            f"<section id='{anchor_id(row['option'])}'>"
            f"<h2>{escape(row['option'])}</h2>"
            f"<p><strong>Documented coverage:</strong> {escape(reference['coverage'])}. "
            f"<a href='{escape(reference['source'])}'>Open source markdown</a>.</p>"
            f"<h3>Documented subtests</h3><ul>{subtest_items}</ul>"
            "<h3>Case dimensions</h3>"
            f"<ul>{case_items}</ul>"
            "<h3>Timed-run cases</h3>"
            f"<ul>{timed_items}</ul>"
            f"<p>Worker wall time: <strong>{worker_timings.get(row['option'], 0.0):.2f}s</strong>.</p>"
            "<p><a href='#top'>Back to table</a></p>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Proxy Timing And Performance Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #17202a; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .note {{ max-width: 980px; line-height: 1.45; color: #3f4b57; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 14px; }}
    th, td {{ border: 1px solid #d7dde3; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #f2f5f8; text-align: left; position: sticky; top: 0; }}
    tr:nth-child(even) td {{ background: #fbfcfd; }}
    .num {{ text-align: right; white-space: nowrap; }}
    code {{ background: #f2f5f8; padding: 1px 4px; border-radius: 3px; }}
    section {{ margin-top: 34px; max-width: 1000px; }}
    section li {{ margin: 4px 0; }}
    a {{ color: #195f9c; }}
  </style>
</head>
<body>
  <h1 id="top">Proxy Timing And Performance Summary</h1>
  <p class="note">
    Generated by <code>ProxyTimingBenchmark.py</code>. Documented performance
    comes from the product markdown summaries; timed-run performance comes from
    a reduced-budget representative timing pass. Benchmark generation is not
    counted in sample-generation or proxy-training time.
  </p>
  <p class="note">
    Total benchmark wall time: <strong>{total_seconds:.2f}s</strong>.
    Worker processes: <strong>{worker_count}</strong>.
    Minimum timed scenario combinations per option family: <strong>{MIN_TIMED_SCENARIO_COMBINATIONS}</strong>.
  </p>
  <table>
    <thead>
      <tr>
        <th>Option type</th>
        <th>Timed scenario combinations</th>
        <th>Executed valuation states</th>
        <th>Details</th>
        <th>Documented default/best method</th>
        <th>Documented worst max</th>
        <th>Documented avg p99</th>
        <th>Smoke avg max</th>
        <th>Smoke worst max</th>
        <th>Smoke avg p99</th>
        <th>Avg sample gen / fit</th>
        <th>Avg proxy train / fit</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  <p class="note">
    Cells marked "timing rerun" mean the original option-specific summary did
    not store an aggregate p99 statistic, so this dashboard fills the value
    from the current representative timing pass.
  </p>
  {''.join(detail_sections)}
</body>
</html>
"""


def main():
    start = time.perf_counter()
    requested_workers = int(os.environ.get("PROXY_BENCHMARK_WORKERS", DEFAULT_WORKERS))
    max_workers = len(benchmark_registry())
    worker_count = max(1, min(requested_workers, max_workers))
    rows, worker_timings = run_benchmarks(worker_count)
    total_seconds = time.perf_counter() - start
    summary = aggregate(rows)
    too_small = {
        row["option"]: row["fits"]
        for row in summary
        if row["fits"] < MIN_TIMED_SCENARIO_COMBINATIONS
    }
    if too_small:
        raise AssertionError(f"Timed scenario combinations below 100: {too_small}")
    too_few_states = {
        row["option"]: row["valuation_cases"]
        for row in summary
        if row["valuation_cases"] < MIN_EXECUTED_VALUATION_CASES
    }
    if too_few_states:
        raise AssertionError(f"Executed valuation states below 100: {too_few_states}")
    text = markdown_table(summary, total_seconds, worker_count, worker_timings)
    html = html_table(summary, total_seconds, worker_count, worker_timings)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="ascii") as handle:
        handle.write(text)
    with open(OUTPUT_HTML, "w", encoding="ascii") as handle:
        handle.write(html)
    print(text)
    print(f"summary written to: {OUTPUT_MD}")
    print(f"html written to: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
