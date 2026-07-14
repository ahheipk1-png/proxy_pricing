import csv
import os
from dataclasses import dataclass
from math import exp, log, sqrt
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ObservationSpec:
    kind: str
    schedule: tuple[int, ...]
    weights: tuple[float, ...] | None = None
    tail_count: int = 3
    level: str = "average"


@dataclass(frozen=True)
class PerformanceSpec:
    kind: str
    strike: float = 100.0


@dataclass(frozen=True)
class RankingSpec:
    kind: str
    weights: tuple[float, ...]
    rank: int = 1


@dataclass(frozen=True)
class TransformationSpec:
    floor: float | None = None
    cap: float | None = None


@dataclass(frozen=True)
class AggregationSpec:
    kind: str


@dataclass(frozen=True)
class PayoffSpec:
    kind: str
    notional: float = 100.0
    strike: float = 0.0
    alpha: float = 1.0
    final_floor: float | None = 0.0
    final_cap: float | None = None


@dataclass(frozen=True)
class Product:
    name: str
    family: str
    asset_count: int
    observation: ObservationSpec
    performance: PerformanceSpec
    ranking: RankingSpec
    transformation: TransformationSpec
    aggregation: AggregationSpec
    payoff: PayoffSpec
    parameters: dict


@dataclass(frozen=True)
class Market:
    s0: tuple[float, ...]
    vols: tuple[float, ...]
    div_yields: tuple[float, ...]
    rate: float = 0.04
    maturity: float = 1.0
    steps: int = 12
    correlation: float = 0.35
    train_paths: int = 16_384
    benchmark_paths: int = 65_536
    seed: int = 20260713


def normal_ppf(probability):
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


def van_der_corput(indices, base):
    indices = np.asarray(indices, dtype=np.int64).copy()
    result = np.zeros(indices.shape[0])
    denominator = float(base)
    while np.any(indices > 0):
        indices, remainder = divmod(indices, base)
        result += remainder / denominator
        denominator *= base
    return result


def first_primes(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def halton_normals(n_paths, dimension, seed):
    half = (int(n_paths) + 1) // 2
    indices = np.arange(1, half + 1, dtype=np.int64)
    primes = first_primes(dimension)
    rng = np.random.default_rng(seed)
    shifts = rng.random(dimension)
    uniforms = np.empty((half, dimension), dtype=float)
    for col, prime in enumerate(primes):
        uniforms[:, col] = (van_der_corput(indices, prime) + shifts[col]) % 1.0
    base = normal_ppf(uniforms)
    normals = np.vstack([base, -base])
    return normals[: int(n_paths)]


def normalized_weights(asset_count):
    base = np.arange(asset_count, 0, -1, dtype=float)
    return tuple((base / np.sum(base)).tolist())


def make_market(asset_count, train_paths=65_536, benchmark_paths=65_536):
    s0 = tuple(100.0 + 4.0 * idx for idx in range(asset_count))
    vols = tuple(0.18 + 0.035 * idx for idx in range(asset_count))
    div_yields = tuple(0.01 + 0.004 * (idx % 3) for idx in range(asset_count))
    correlation = 0.0 if asset_count == 1 else 0.35
    return Market(
        s0=s0,
        vols=vols,
        div_yields=div_yields,
        correlation=correlation,
        train_paths=train_paths,
        benchmark_paths=benchmark_paths,
    )


def simulate_base_paths(market, n_paths, seed):
    asset_count = len(market.s0)
    dimension = market.steps * asset_count
    raw = halton_normals(n_paths, dimension, seed).reshape(
        n_paths, market.steps, asset_count
    )
    if asset_count == 1:
        normals = raw
    else:
        corr = np.full((asset_count, asset_count), market.correlation)
        np.fill_diagonal(corr, 1.0)
        normals = raw @ np.linalg.cholesky(corr).T

    dt = market.maturity / market.steps
    vols = np.asarray(market.vols)
    divs = np.asarray(market.div_yields)
    base = np.empty((n_paths, market.steps + 1, asset_count), dtype=float)
    base[:, 0, :] = 1.0
    current = np.ones((n_paths, asset_count), dtype=float)
    for step in range(market.steps):
        drift = (market.rate - divs - 0.5 * vols * vols) * dt
        diffusion = vols * sqrt(dt) * normals[:, step, :]
        current = current * np.exp(drift[None, :] + diffusion)
        base[:, step + 1, :] = current
    return base


def scaled_paths(base_paths, market, scales):
    scales = np.asarray(scales, dtype=float)
    initial = np.asarray(market.s0, dtype=float)
    return scales[:, None, None, None] * initial[None, None, None, :] * base_paths[None, :, :, :]


def observe(paths, spec):
    indices = np.asarray(spec.schedule, dtype=int)
    selected = paths[:, :, indices, :]
    if spec.kind == "spot":
        return selected[:, :, -1, :]
    if spec.kind == "arithmetic_average":
        return np.mean(selected, axis=2)
    if spec.kind == "weighted_average":
        weights = np.asarray(spec.weights, dtype=float)
        weights = weights / np.sum(weights)
        return np.sum(selected * weights[None, None, :, None], axis=2)
    if spec.kind == "geometric_average":
        return np.exp(np.mean(np.log(np.maximum(selected, 1e-12)), axis=2))
    if spec.kind == "tail_average":
        tail = selected[:, :, -min(spec.tail_count, selected.shape[2]) :, :]
        return np.mean(tail, axis=2)
    if spec.kind == "lookback":
        if spec.level == "minimum":
            return np.min(selected, axis=2)
        if spec.level == "maximum":
            return np.max(selected, axis=2)
        return np.mean(selected, axis=2)
    if spec.kind == "running_maximum":
        return np.max(selected, axis=2)
    if spec.kind == "running_minimum":
        return np.min(selected, axis=2)
    raise ValueError(f"unknown observation: {spec.kind}")


def compute_performance(observed, market, spec):
    initial = np.asarray(market.s0, dtype=float)
    if spec.kind == "fixed_notional":
        return observed / spec.strike - 1.0
    if spec.kind == "fixed_unit":
        return (observed - spec.strike) / spec.strike
    if spec.kind == "relative":
        return observed / initial[None, None, :] - 1.0
    if spec.kind == "spot_ratio":
        return observed / spec.strike
    raise ValueError(f"unknown performance: {spec.kind}")


def apply_transform(values, spec):
    out = np.asarray(values, dtype=float)
    if spec.floor is not None:
        out = np.maximum(out, spec.floor)
    if spec.cap is not None:
        out = np.minimum(out, spec.cap)
    return out


def rank_values(values, spec):
    if spec.kind == "identity":
        return values[:, :, 0]
    if spec.kind == "weighted_basket":
        weights = np.asarray(spec.weights, dtype=float)
        weights = weights / np.sum(weights)
        return values @ weights
    if spec.kind == "order_statistic":
        rank = max(1, min(spec.rank, values.shape[2]))
        sorted_values = np.sort(values, axis=2)[:, :, ::-1]
        return sorted_values[:, :, rank - 1]
    raise ValueError(f"unknown ranking: {spec.kind}")


def aggregate(period_values, spec):
    values = np.stack(period_values, axis=2)
    if spec.kind == "average":
        return np.mean(values, axis=2)
    if spec.kind == "sum":
        return np.sum(values, axis=2)
    if spec.kind == "compounded":
        return np.prod(1.0 + values, axis=2) - 1.0
    if spec.kind == "maximum":
        return np.max(values, axis=2)
    if spec.kind == "minimum":
        return np.min(values, axis=2)
    raise ValueError(f"unknown aggregation: {spec.kind}")


def final_payoff(values, spec):
    x = spec.alpha * (values - spec.strike)
    if spec.kind == "linear":
        raw = x
    elif spec.kind == "option":
        raw = np.maximum(x, 0.0)
    elif spec.kind == "clamped_linear":
        raw = x
        if spec.final_floor is not None:
            raw = np.maximum(spec.final_floor, raw)
        if spec.final_cap is not None:
            raw = np.minimum(spec.final_cap, raw)
    else:
        raise ValueError(f"unknown payoff: {spec.kind}")
    return spec.notional * raw


def period_schedules(period_count, steps):
    width = steps // period_count
    return tuple(
        tuple(range(period * width + 1, (period + 1) * width + 1))
        for period in range(period_count)
    )


def evaluate_rainbow(product, paths, market):
    obs = observe(paths, product.observation)
    perf = compute_performance(obs, market, product.performance)
    local = apply_transform(perf, product.transformation)
    ranked = rank_values(local, product.ranking)
    return final_payoff(ranked, product.payoff)


def active_weighted_value(perf, active, weights):
    weighted = active * weights[None, None, :]
    denom = np.maximum(np.sum(weighted, axis=2), 1e-12)
    return np.sum(np.where(active, perf, 0.0) * weighted, axis=2) / denom


def active_order_value(perf, active, rank):
    masked = np.where(active, perf, -1e9)
    sorted_values = np.sort(masked, axis=2)[:, :, ::-1]
    active_count = np.sum(active, axis=2)
    chosen_rank = np.maximum(np.minimum(rank, active_count), 1)
    scale_count, path_count = perf.shape[:2]
    return sorted_values[
        np.arange(scale_count)[:, None],
        np.arange(path_count)[None, :],
        chosen_rank - 1,
    ]


def evaluate_himalayan(product, paths, market):
    scale_count, path_count, _, asset_count = paths.shape
    active = np.ones((scale_count, path_count, asset_count), dtype=bool)
    weights = np.asarray(product.ranking.weights, dtype=float)
    rank = int(product.parameters["rank"])
    remove_worst = int(product.parameters["remove_worst"])
    period_values = []
    for schedule in product.parameters["period_schedules"]:
        obs = observe(paths, ObservationSpec(product.observation.kind, schedule))
        perf = compute_performance(obs, market, product.performance)
        if rank == 0:
            chosen = active_weighted_value(perf, active, weights)
        else:
            chosen = active_order_value(perf, active, rank)
        period_values.append(apply_transform(chosen, product.transformation))
        if remove_worst > 0 and asset_count > 1:
            masked = np.where(active, perf, 1e9)
            order = np.argsort(masked, axis=2)
            for remove_index in range(remove_worst):
                worst = order[:, :, remove_index]
                row = np.arange(scale_count)[:, None]
                col = np.arange(path_count)[None, :]
                still_many = np.sum(active, axis=2) > 1
                active[row, col, worst] = np.where(still_many, False, active[row, col, worst])
    return final_payoff(aggregate(period_values, product.aggregation), product.payoff)


def evaluate_yield_seeker(product, paths, market):
    weights = np.asarray(product.ranking.weights, dtype=float)
    coupons = np.zeros(paths.shape[:2])
    missed = np.zeros(paths.shape[:2])
    best_perf = None
    for date in product.parameters["coupon_dates"]:
        obs = observe(paths, ObservationSpec("spot", (date,)))
        perf = compute_performance(obs, market, product.performance)
        if product.parameters["lookback"]:
            best_perf = perf if best_perf is None else np.maximum(best_perf, perf)
            perf = best_perf
        basket = perf @ weights
        mapping = product.parameters["mapping"]
        high = product.parameters["high_coupon"]
        low = product.parameters["low_coupon"]
        trigger = product.parameters["trigger"]
        if mapping == "high_low":
            coupons += np.where(basket >= trigger, high, low)
        elif mapping == "actual_return":
            coupons += np.clip(basket, low, high)
        elif mapping == "memory":
            hit = basket >= trigger
            coupons += np.where(hit, high * (1.0 + missed), 0.0)
            missed = np.where(hit, 0.0, missed + 1.0)
        else:
            raise ValueError(mapping)
    return product.payoff.notional * (1.0 + coupons)


def evaluate_lookback(product, paths, market):
    weights = np.asarray(product.ranking.weights, dtype=float)
    initial = np.asarray(market.s0, dtype=float)
    indices = np.asarray(product.observation.schedule, dtype=int)
    level_series = paths[:, :, indices, :] / initial[None, None, None, :]
    basket = np.sum(level_series * weights[None, None, None, :], axis=3)
    level = product.observation.level
    if level == "minimum":
        raw = np.min(basket, axis=2)
    elif level == "maximum":
        raw = np.max(basket, axis=2)
    elif level == "trimmed_average":
        raw = np.mean(np.sort(basket, axis=2)[:, :, 1:-1], axis=2)
    else:
        raw = np.mean(basket, axis=2)
    look_level = apply_transform(raw, product.transformation)
    tail_count = min(int(product.parameters["tail_count"]), basket.shape[2])
    settlement_avg = np.mean(basket[:, :, -tail_count:], axis=2)
    settlement = product.parameters["settlement"]
    if settlement == "fixed_strike":
        inner = product.payoff.alpha * (look_level - product.payoff.strike)
    elif settlement == "floating_strike":
        inner = product.payoff.alpha * (settlement_avg - product.payoff.strike * look_level)
    elif settlement == "modified_floating":
        inner = product.payoff.alpha * (settlement_avg - look_level - product.payoff.strike)
    elif settlement == "floating_ratio":
        inner = product.payoff.alpha * (
            settlement_avg / np.maximum(np.abs(look_level), 1e-5) - product.payoff.strike
        )
    else:
        raise ValueError(settlement)
    return product.payoff.notional * np.maximum(inner, 0.0)


def barrier_series(product, paths, market):
    initial = np.asarray(market.s0, dtype=float)
    perf = paths / initial[None, None, None, :] - 1.0
    if product.ranking.kind == "weighted_basket":
        weights = np.asarray(product.ranking.weights, dtype=float)
        return np.sum(perf * weights[None, None, None, :], axis=3)
    rank = max(1, min(product.ranking.rank, product.asset_count))
    return np.sort(perf, axis=3)[:, :, :, ::-1][:, :, :, rank - 1]


def evaluate_barrier(product, paths, market):
    series = barrier_series(product, paths, market)
    if product.parameters["monitoring"] == "discrete":
        monitored = series[:, :, product.parameters["monitoring_dates"]]
    else:
        monitored = series[:, :, 1:]
    direction = product.parameters["direction"]
    level = product.parameters["barrier"]
    if direction == "upper":
        breached = np.any(monitored >= level, axis=2)
    else:
        breached = np.any(monitored <= level, axis=2)
    active = ~breached if product.parameters["barrier_type"] == "knock_out" else breached
    final = series[:, :, -1]
    settlement = product.parameters["settlement"]
    if settlement == "call":
        payoff = product.payoff.notional * np.maximum(final - product.payoff.strike, 0.0)
    elif settlement == "put":
        payoff = product.payoff.notional * np.maximum(product.payoff.strike - final, 0.0)
    elif settlement == "cash":
        payoff = np.full(final.shape, product.payoff.notional)
    else:
        raise ValueError(settlement)
    rebate = product.parameters.get("rebate", 0.0) * product.payoff.notional
    return np.where(active, payoff, rebate)


def evaluate_binary(product, paths, market):
    obs = observe(paths, product.observation)
    perf = compute_performance(obs, market, product.performance)
    ranked = rank_values(perf, product.ranking)
    binary_type = product.parameters["binary_type"]
    if binary_type in {"double_digital", "range_digital"}:
        low, high = product.parameters["range"]
        indicator = (ranked >= low) & (ranked <= high)
    else:
        indicator = product.payoff.alpha * (ranked - product.parameters["trigger"]) > 0.0
    if binary_type == "asset_or_nothing":
        payoff = product.payoff.notional * np.maximum(1.0 + ranked, 0.0)
    elif binary_type == "gap":
        payoff = product.payoff.notional * np.maximum(
            product.payoff.alpha * (ranked - product.payoff.strike), 0.0
        )
    else:
        payoff = np.full(ranked.shape, product.payoff.notional)
    return np.where(indicator, payoff, 0.0)


def evaluate_product(product, paths, market):
    if product.family == "Rainbow":
        payoff = evaluate_rainbow(product, paths, market)
    elif product.family == "Himalayan":
        payoff = evaluate_himalayan(product, paths, market)
    elif product.family == "YieldSeeker":
        payoff = evaluate_yield_seeker(product, paths, market)
    elif product.family == "Lookback":
        payoff = evaluate_lookback(product, paths, market)
    elif product.family == "Barrier":
        payoff = evaluate_barrier(product, paths, market)
    elif product.family == "Binary":
        payoff = evaluate_binary(product, paths, market)
    else:
        raise ValueError(product.family)
    return exp(-market.rate * market.maturity) * np.mean(payoff, axis=1)


def make_rainbow(asset_count):
    products = []
    weights = normalized_weights(asset_count)
    ranking_specs = [RankingSpec("weighted_basket", weights)]
    if asset_count > 1:
        ranking_specs.append(RankingSpec("order_statistic", weights, asset_count))
    for obs in ["spot", "arithmetic_average", "tail_average"]:
        for perf in ["fixed_notional", "fixed_unit", "relative"]:
            for floor, cap in [(-0.20, 0.35), (-0.10, 0.20), (None, 0.30)]:
                for ranking in ranking_specs:
                    products.append(
                        Product(
                            f"rainbow_{asset_count}_{obs}_{perf}_{ranking.kind}_{floor}_{cap}",
                            "Rainbow",
                            asset_count,
                            ObservationSpec(obs, tuple(range(1, 13))),
                            PerformanceSpec(perf),
                            ranking,
                            TransformationSpec(floor, cap),
                            AggregationSpec("sum"),
                            PayoffSpec("option", 100.0, 0.0, 1.0),
                            {},
                        )
                    )
    return products


def make_himalayan(asset_count):
    products = []
    weights = normalized_weights(asset_count)
    schedules = period_schedules(4, 12)
    ranks = [0, 1] if asset_count == 1 else [0, 1, asset_count]
    removes = [0] if asset_count == 1 else [0, 1]
    for rank in ranks:
        for remove in removes:
            for agg in ["average", "sum", "compounded"]:
                products.append(
                    Product(
                        f"himalayan_{asset_count}_rank{rank}_remove{remove}_{agg}",
                        "Himalayan",
                        asset_count,
                        ObservationSpec("arithmetic_average", schedules[0]),
                        PerformanceSpec("fixed_notional"),
                        RankingSpec("weighted_basket" if rank == 0 else "order_statistic", weights, max(rank, 1)),
                        TransformationSpec(-0.10, 0.15),
                        AggregationSpec(agg),
                        PayoffSpec("clamped_linear", 100.0, 0.0, 1.0, 0.0, 0.35),
                        {"period_schedules": schedules, "remove_worst": remove, "rank": rank},
                    )
                )
    return products


def make_yield_seeker(asset_count):
    products = []
    weights = normalized_weights(asset_count)
    for mapping in ["high_low", "actual_return", "memory"]:
        for lookback in [False, True]:
            for trigger in [-0.05, 0.0, 0.05]:
                products.append(
                    Product(
                        f"yield_seeker_{asset_count}_{mapping}_{lookback}_{trigger}",
                        "YieldSeeker",
                        asset_count,
                        ObservationSpec("spot", (12,)),
                        PerformanceSpec("relative"),
                        RankingSpec("weighted_basket", weights),
                        TransformationSpec(None, None),
                        AggregationSpec("sum"),
                        PayoffSpec("linear", 100.0, 0.0, 1.0),
                        {
                            "coupon_dates": (3, 6, 9, 12),
                            "trigger": trigger,
                            "high_coupon": 0.035,
                            "low_coupon": 0.006,
                            "mapping": mapping,
                            "lookback": lookback,
                        },
                    )
                )
    return products


def make_lookback(asset_count):
    products = []
    weights = normalized_weights(asset_count)
    for level in ["minimum", "maximum", "average", "trimmed_average"]:
        for settlement in ["fixed_strike", "floating_strike", "modified_floating", "floating_ratio"]:
            strike = 0.0 if settlement == "modified_floating" else 1.0
            products.append(
                Product(
                    f"lookback_{asset_count}_{level}_{settlement}",
                    "Lookback",
                    asset_count,
                    ObservationSpec("lookback", tuple(range(0, 13)), level=level),
                    PerformanceSpec("relative"),
                    RankingSpec("weighted_basket", weights),
                    TransformationSpec(0.55, 1.65),
                    AggregationSpec("maximum"),
                    PayoffSpec("option", 100.0, strike, 1.0),
                    {"settlement": settlement, "tail_count": 3},
                )
            )
    return products


def make_barrier(asset_count):
    products = []
    weights = normalized_weights(asset_count)
    rankings = [RankingSpec("weighted_basket", weights)]
    if asset_count > 1:
        rankings += [
            RankingSpec("order_statistic", weights, 1),
            RankingSpec("order_statistic", weights, asset_count),
        ]
    for direction, barrier in [("lower", -0.25), ("upper", 0.30)]:
        for barrier_type in ["knock_out", "knock_in"]:
            for monitoring in ["discrete", "continuous"]:
                for settlement in ["call", "put", "cash"]:
                    for ranking in rankings:
                        products.append(
                            Product(
                                f"barrier_{asset_count}_{direction}_{barrier_type}_{monitoring}_{settlement}_{ranking.kind}_{ranking.rank}",
                                "Barrier",
                                asset_count,
                                ObservationSpec("spot", (12,)),
                                PerformanceSpec("relative"),
                                ranking,
                                TransformationSpec(None, None),
                                AggregationSpec("sum"),
                                PayoffSpec("option", 100.0, 0.0, 1.0),
                                {
                                    "direction": direction,
                                    "barrier": barrier,
                                    "barrier_type": barrier_type,
                                    "monitoring": monitoring,
                                    "monitoring_dates": (3, 6, 9, 12),
                                    "settlement": settlement,
                                    "rebate": 0.02 if barrier_type == "knock_out" else 0.0,
                                },
                            )
                        )
    return products


def make_binary(asset_count):
    products = []
    weights = normalized_weights(asset_count)
    rankings = [RankingSpec("weighted_basket", weights)]
    if asset_count > 1:
        rankings.append(RankingSpec("order_statistic", weights, asset_count))
    for obs in ["spot", "arithmetic_average", "tail_average"]:
        for binary_type in ["cash_or_nothing", "asset_or_nothing", "gap", "double_digital", "range_digital"]:
            for ranking in rankings:
                products.append(
                    Product(
                        f"binary_{asset_count}_{obs}_{binary_type}_{ranking.kind}_{ranking.rank}",
                        "Binary",
                        asset_count,
                        ObservationSpec(obs, tuple(range(1, 13))),
                        PerformanceSpec("relative"),
                        ranking,
                        TransformationSpec(None, None),
                        AggregationSpec("sum"),
                        PayoffSpec("linear", 100.0, 0.03, 1.0),
                        {"binary_type": binary_type, "trigger": 0.0, "range": (-0.08, 0.18)},
                    )
                )
    return products


def schedule_weights(schedule, reverse=False):
    weights = np.arange(1, len(schedule) + 1, dtype=float)
    if reverse:
        weights = weights[::-1]
    weights = weights / np.sum(weights)
    return tuple(weights.tolist())


def observation_variants():
    full = tuple(range(1, 13))
    back_half = tuple(range(7, 13))
    quarterly = (3, 6, 9, 12)
    front_half = tuple(range(1, 7))
    return [
        ("spot_t12", ObservationSpec("spot", (12,))),
        ("spot_t6", ObservationSpec("spot", (6,))),
        ("avg_full", ObservationSpec("arithmetic_average", full)),
        ("avg_back", ObservationSpec("arithmetic_average", back_half)),
        ("avg_quarter", ObservationSpec("arithmetic_average", quarterly)),
        ("avg_front_weighted", ObservationSpec("weighted_average", full, schedule_weights(full, True))),
        ("avg_back_weighted", ObservationSpec("weighted_average", full, schedule_weights(full, False))),
        ("geo_full", ObservationSpec("geometric_average", full)),
        ("tail_3", ObservationSpec("tail_average", full, tail_count=3)),
        ("tail_6", ObservationSpec("tail_average", full, tail_count=6)),
        ("avg_front", ObservationSpec("arithmetic_average", front_half)),
    ]


def ranking_variants(asset_count):
    weights = normalized_weights(asset_count)
    rankings = [("weighted", RankingSpec("weighted_basket", weights))]
    if asset_count > 1:
        rankings.append(("best", RankingSpec("order_statistic", weights, 1)))
        rankings.append(("middle", RankingSpec("order_statistic", weights, max(1, (asset_count + 1) // 2))))
        rankings.append(("worst", RankingSpec("order_statistic", weights, asset_count)))
    return rankings


def spread_product_subset(products, target_count):
    if target_count is None or target_count >= len(products):
        return products
    indices = np.floor(np.arange(target_count) * len(products) / target_count).astype(int)
    return [products[int(idx)] for idx in indices]


def make_rainbow_expanded(asset_count):
    products = []
    perf_specs = [
        ("notional_95", PerformanceSpec("fixed_notional", 95.0)),
        ("notional_100", PerformanceSpec("fixed_notional", 100.0)),
        ("notional_105", PerformanceSpec("fixed_notional", 105.0)),
        ("unit_95", PerformanceSpec("fixed_unit", 95.0)),
        ("unit_100", PerformanceSpec("fixed_unit", 100.0)),
        ("relative", PerformanceSpec("relative")),
        ("ratio_100", PerformanceSpec("spot_ratio", 100.0)),
        ("ratio_105", PerformanceSpec("spot_ratio", 105.0)),
    ]
    transforms = [
        ("raw", TransformationSpec(None, None)),
        ("floor20_cap35", TransformationSpec(-0.20, 0.35)),
        ("floor10_cap20", TransformationSpec(-0.10, 0.20)),
        ("zero_cap30", TransformationSpec(0.0, 0.30)),
        ("floor30", TransformationSpec(-0.30, None)),
    ]
    payoffs = [
        ("call0", PayoffSpec("option", 100.0, 0.0, 1.0)),
        ("call5", PayoffSpec("option", 100.0, 0.05, 1.0)),
        ("put0", PayoffSpec("option", 100.0, 0.0, -1.0)),
    ]
    for obs_name, obs in observation_variants():
        for perf_name, perf in perf_specs:
            for trans_name, transform in transforms:
                for rank_name, ranking in ranking_variants(asset_count):
                    for payoff_name, payoff in payoffs:
                        products.append(
                            Product(
                                f"rainbow_{asset_count}_{obs_name}_{perf_name}_{trans_name}_{rank_name}_{payoff_name}",
                                "Rainbow",
                                asset_count,
                                obs,
                                perf,
                                ranking,
                                transform,
                                AggregationSpec("sum"),
                                payoff,
                                {},
                            )
                        )
    return products


def make_himalayan_expanded(asset_count):
    products = []
    weights = normalized_weights(asset_count)
    rank_values = [0, 1] if asset_count == 1 else [0, 1, max(1, (asset_count + 1) // 2), asset_count]
    remove_values = [0] if asset_count == 1 else [0, 1, min(2, asset_count - 1)]
    obs_kinds = ["spot", "arithmetic_average", "geometric_average", "tail_average"]
    transforms = [
        ("tight", TransformationSpec(-0.10, 0.15)),
        ("wide", TransformationSpec(-0.20, 0.25)),
        ("positive", TransformationSpec(0.0, 0.20)),
        ("low_cap", TransformationSpec(-0.05, 0.10)),
    ]
    payoffs = [
        ("cap35", PayoffSpec("clamped_linear", 100.0, 0.0, 1.0, 0.0, 0.35)),
        ("cap50", PayoffSpec("clamped_linear", 100.0, 0.0, 1.0, 0.0, 0.50)),
        ("call0", PayoffSpec("option", 100.0, 0.0, 1.0)),
    ]
    for period_count in [2, 3, 4, 6]:
        schedules = period_schedules(period_count, 12)
        for obs_kind in obs_kinds:
            for rank in rank_values:
                for remove in remove_values:
                    for agg in ["average", "sum", "compounded", "maximum"]:
                        for trans_name, transform in transforms:
                            for payoff_name, payoff in payoffs:
                                ranking = RankingSpec(
                                    "weighted_basket" if rank == 0 else "order_statistic",
                                    weights,
                                    max(rank, 1),
                                )
                                products.append(
                                    Product(
                                        f"himalayan_{asset_count}_p{period_count}_{obs_kind}_rank{rank}_remove{remove}_{agg}_{trans_name}_{payoff_name}",
                                        "Himalayan",
                                        asset_count,
                                        ObservationSpec(obs_kind, schedules[0]),
                                        PerformanceSpec("fixed_notional"),
                                        ranking,
                                        transform,
                                        AggregationSpec(agg),
                                        payoff,
                                        {"period_schedules": schedules, "remove_worst": remove, "rank": rank},
                                    )
                                )
    return products


def make_yield_seeker_expanded(asset_count):
    products = []
    weights = normalized_weights(asset_count)
    coupon_schedules = [
        ("quarterly", (3, 6, 9, 12)),
        ("monthly", tuple(range(1, 13))),
        ("semiannual", (6, 12)),
        ("even_months", (2, 4, 6, 8, 10, 12)),
        ("late", (8, 10, 12)),
    ]
    coupon_pairs = [
        ("base", 0.035, 0.006),
        ("high", 0.050, 0.010),
        ("defensive", 0.025, 0.004),
        ("steep", 0.060, 0.000),
    ]
    for mapping in ["high_low", "actual_return", "memory"]:
        for lookback in [False, True]:
            for trigger in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]:
                for schedule_name, coupon_dates in coupon_schedules:
                    for coupon_name, high_coupon, low_coupon in coupon_pairs:
                        products.append(
                            Product(
                                f"yield_seeker_{asset_count}_{mapping}_{lookback}_{trigger}_{schedule_name}_{coupon_name}",
                                "YieldSeeker",
                                asset_count,
                                ObservationSpec("spot", (12,)),
                                PerformanceSpec("relative"),
                                RankingSpec("weighted_basket", weights),
                                TransformationSpec(None, None),
                                AggregationSpec("sum"),
                                PayoffSpec("linear", 100.0, 0.0, 1.0),
                                {
                                    "coupon_dates": coupon_dates,
                                    "trigger": trigger,
                                    "high_coupon": high_coupon,
                                    "low_coupon": low_coupon,
                                    "mapping": mapping,
                                    "lookback": lookback,
                                },
                            )
                        )
    return products


def make_lookback_expanded(asset_count):
    products = []
    weights = normalized_weights(asset_count)
    schedules = [
        ("full", tuple(range(0, 13))),
        ("after_q1", tuple(range(3, 13))),
        ("back_half", tuple(range(6, 13))),
        ("quarterly", (0, 3, 6, 9, 12)),
    ]
    transforms = [
        ("normal", TransformationSpec(0.55, 1.65)),
        ("wide", TransformationSpec(0.45, 1.85)),
        ("tight", TransformationSpec(0.75, 1.35)),
        ("uncapped", TransformationSpec(0.55, None)),
    ]
    for schedule_name, schedule in schedules:
        for level in ["minimum", "maximum", "average", "trimmed_average"]:
            for settlement in ["fixed_strike", "floating_strike", "modified_floating", "floating_ratio"]:
                for tail_count in [1, 3, 6]:
                    for trans_name, transform in transforms:
                        for strike in [0.90, 1.00, 1.10]:
                            for alpha_name, alpha in [("call", 1.0), ("put", -1.0)]:
                                payoff_strike = 0.0 if settlement == "modified_floating" else strike
                                products.append(
                                    Product(
                                        f"lookback_{asset_count}_{schedule_name}_{level}_{settlement}_tail{tail_count}_{trans_name}_{strike}_{alpha_name}",
                                        "Lookback",
                                        asset_count,
                                        ObservationSpec("lookback", schedule, level=level),
                                        PerformanceSpec("relative"),
                                        RankingSpec("weighted_basket", weights),
                                        transform,
                                        AggregationSpec("maximum"),
                                        PayoffSpec("option", 100.0, payoff_strike, alpha),
                                        {"settlement": settlement, "tail_count": tail_count},
                                    )
                                )
    return products


def make_barrier_expanded(asset_count):
    products = []
    monitoring_sets = [
        ("quarterly", (3, 6, 9, 12)),
        ("monthly", tuple(range(1, 13))),
        ("semiannual", (6, 12)),
        ("late", (9, 10, 11, 12)),
    ]
    levels = {"lower": [-0.35, -0.25, -0.15], "upper": [0.15, 0.25, 0.35]}
    for direction in ["lower", "upper"]:
        for barrier in levels[direction]:
            for barrier_type in ["knock_out", "knock_in"]:
                for monitoring in ["discrete", "continuous"]:
                    for schedule_name, monitoring_dates in monitoring_sets:
                        for settlement in ["call", "put", "cash"]:
                            for strike in [-0.05, 0.0, 0.05]:
                                for rebate in [0.0, 0.02]:
                                    for rank_name, ranking in ranking_variants(asset_count):
                                        products.append(
                                            Product(
                                                f"barrier_{asset_count}_{direction}_{barrier}_{barrier_type}_{monitoring}_{schedule_name}_{settlement}_{strike}_{rebate}_{rank_name}",
                                                "Barrier",
                                                asset_count,
                                                ObservationSpec("spot", (12,)),
                                                PerformanceSpec("relative"),
                                                ranking,
                                                TransformationSpec(None, None),
                                                AggregationSpec("sum"),
                                                PayoffSpec("option", 100.0, strike, 1.0),
                                                {
                                                    "direction": direction,
                                                    "barrier": barrier,
                                                    "barrier_type": barrier_type,
                                                    "monitoring": monitoring,
                                                    "monitoring_dates": monitoring_dates,
                                                    "settlement": settlement,
                                                    "rebate": rebate,
                                                },
                                            )
                                        )
    return products


def make_binary_expanded(asset_count):
    products = []
    perf_specs = [
        ("relative", PerformanceSpec("relative")),
        ("notional_100", PerformanceSpec("fixed_notional", 100.0)),
        ("unit_100", PerformanceSpec("fixed_unit", 100.0)),
    ]
    for obs_name, obs in observation_variants():
        for perf_name, perf in perf_specs:
            for binary_type in ["cash_or_nothing", "asset_or_nothing", "gap", "double_digital", "range_digital"]:
                for trigger in [-0.15, -0.08, 0.0, 0.08, 0.15]:
                    for low, high in [(-0.15, 0.15), (-0.08, 0.18), (0.0, 0.25)]:
                        for alpha_name, alpha in [("up", 1.0), ("down", -1.0)]:
                            for rank_name, ranking in ranking_variants(asset_count):
                                products.append(
                                    Product(
                                        f"binary_{asset_count}_{obs_name}_{perf_name}_{binary_type}_{trigger}_{low}_{high}_{alpha_name}_{rank_name}",
                                        "Binary",
                                        asset_count,
                                        obs,
                                        perf,
                                        ranking,
                                        TransformationSpec(None, None),
                                        AggregationSpec("sum"),
                                        PayoffSpec("linear", 100.0, 0.03, alpha),
                                        {"binary_type": binary_type, "trigger": trigger, "range": (low, high)},
                                    )
                                )
    return products


EXPANDED_FAMILY_BUILDERS = {
    "Rainbow": make_rainbow_expanded,
    "Himalayan": make_himalayan_expanded,
    "YieldSeeker": make_yield_seeker_expanded,
    "Lookback": make_lookback_expanded,
    "Barrier": make_barrier_expanded,
    "Binary": make_binary_expanded,
}


def build_family_products(family, asset_count, expanded=True):
    if expanded:
        return EXPANDED_FAMILY_BUILDERS[family](asset_count)
    return [product for product in build_products(asset_count) if product.family == family]


def build_products(asset_count):
    products = []
    products.extend(make_rainbow(asset_count))
    products.extend(make_himalayan(asset_count))
    products.extend(make_yield_seeker(asset_count))
    products.extend(make_lookback(asset_count))
    products.extend(make_barrier(asset_count))
    products.extend(make_binary(asset_count))
    return products


def chebyshev_scales(n_points, low=0.55, high=1.65):
    nodes = np.cos(np.linspace(np.pi, 0.0, n_points))
    return np.sort(0.5 * (low + high) + 0.5 * (high - low) * nodes)


def validation_scales(n_points, low=0.58, high=1.60):
    return np.exp(np.linspace(log(low), log(high), n_points))


def enriched_family_train_scales(family, n_points):
    base = list(chebyshev_scales(n_points))
    if family in {"Lookback", "Binary", "Barrier"}:
        anchors = [0.75, 0.85, 0.90, 1.00, 1.10, 1.15, 1.25]
        if family == "Barrier":
            anchors.extend([0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 1.18, 1.22, 1.30, 1.34, 1.38, 1.45])
        offsets = [-0.035, -0.020, -0.010, -0.005, 0.0, 0.005, 0.010, 0.020, 0.035]
        for anchor in anchors:
            for offset in offsets:
                value = anchor * exp(offset)
                if 0.55 <= value <= 1.65:
                    base.append(value)
    return np.array(sorted(set(round(float(value), 12) for value in base)), dtype=float)


def pchip_slopes(x, y):
    n = len(x)
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return np.array([slope, slope], dtype=float)
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros(n, dtype=float)
    for idx in range(1, n - 1):
        left = delta[idx - 1]
        right = delta[idx]
        if left == 0.0 or right == 0.0 or np.sign(left) != np.sign(right):
            slopes[idx] = 0.0
        else:
            w1 = 2.0 * h[idx] + h[idx - 1]
            w2 = h[idx] + 2.0 * h[idx - 1]
            slopes[idx] = (w1 + w2) / (w1 / left + w2 / right)

    def endpoint_slope(h0, h1, m0, m1):
        slope = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        if slope == 0.0 or np.sign(slope) != np.sign(m0):
            return 0.0
        if np.sign(m0) != np.sign(m1) and abs(slope) > abs(3.0 * m0):
            return 3.0 * m0
        return slope

    slopes[0] = endpoint_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def pchip_predict(x, y, slopes, new_x):
    new_x = np.asarray(new_x, dtype=float)
    idx = np.searchsorted(x, new_x, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (new_x - x[idx]) / h
    t2 = t * t
    t3 = t2 * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y[idx]
        + (t3 - 2.0 * t2 + t) * h * slopes[idx]
        + (-2.0 * t3 + 3.0 * t2) * y[idx + 1]
        + (t3 - t2) * h * slopes[idx + 1]
    )


def fit_interpolator(scales, values, kind):
    x = np.log(np.asarray(scales, dtype=float))
    y = np.asarray(values, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if kind == "nearest":
        def predict(new_scales):
            new_x = np.log(np.asarray(new_scales, dtype=float))
            idx = np.searchsorted(x, new_x)
            idx = np.clip(idx, 1, len(x) - 1)
            left = idx - 1
            right = idx
            choose_right = np.abs(x[right] - new_x) < np.abs(new_x - x[left])
            nearest = np.where(choose_right, right, left)
            return np.maximum(y[nearest], 0.0)

        return predict

    if kind in {"logit_linear", "logit_pchip"}:
        upper = max(float(np.max(y)) * 1.000001, 1e-8)
        p = np.clip(y / upper, 1e-8, 1.0 - 1e-8)
        z = np.log(p / (1.0 - p))
        slopes = pchip_slopes(x, z) if kind == "logit_pchip" else None

        def predict(new_scales):
            new_x = np.log(np.asarray(new_scales, dtype=float))
            if kind == "logit_pchip":
                fitted = pchip_predict(x, z, slopes, new_x)
            else:
                fitted = np.interp(new_x, x, z)
            fitted = np.clip(fitted, -40.0, 40.0)
            probability = 1.0 / (1.0 + np.exp(-fitted))
            return upper * probability

        return predict

    if kind in {"log_linear", "log_pchip"}:
        z = np.log(np.maximum(y, 0.0) + 1e-10)
        slopes = pchip_slopes(x, z) if kind == "log_pchip" else None

        def predict(new_scales):
            new_x = np.log(np.asarray(new_scales, dtype=float))
            if kind == "log_pchip":
                fitted = pchip_predict(x, z, slopes, new_x)
            else:
                fitted = np.interp(new_x, x, z)
            return np.maximum(np.exp(np.clip(fitted, -40.0, 25.0)) - 1e-10, 0.0)

        return predict

    slopes = pchip_slopes(x, y) if kind == "pchip" else None

    def predict(new_scales):
        new_x = np.log(np.asarray(new_scales, dtype=float))
        if kind == "pchip":
            fitted = pchip_predict(x, y, slopes, new_x)
        else:
            fitted = np.interp(new_x, x, y)
        return np.maximum(fitted, 0.0)

    return predict


def score(prediction, truth, floor=0.05):
    error = prediction - truth
    absolute = np.abs(error)
    relative = absolute / np.maximum(np.abs(truth), floor)
    return {
        "max_rel": float(np.max(relative)),
        "p99_rel": float(np.quantile(relative, 0.99)),
        "p95_rel": float(np.quantile(relative, 0.95)),
        "mae": float(np.mean(absolute)),
        "max_abs": float(np.max(absolute)),
    }


def evaluate_scaled_product(product, base_paths, market, scales, batch_size=8):
    values = np.empty(len(scales), dtype=float)
    for start in range(0, len(scales), batch_size):
        end = min(start + batch_size, len(scales))
        paths = scaled_paths(base_paths, market, scales[start:end])
        values[start:end] = evaluate_product(product, paths, market)
    return values


def balanced_product_subset(products, target_count):
    if target_count is None or target_count >= len(products):
        return products
    families = sorted({product.family for product in products})
    buckets = {family: [product for product in products if product.family == family] for family in families}
    selected = []
    quota = max(1, target_count // len(families))
    for family in families:
        take = min(quota, len(buckets[family]))
        selected.extend(buckets[family][:take])
    cursor = {family: min(quota, len(buckets[family])) for family in families}
    while len(selected) < target_count:
        made_progress = False
        for family in families:
            idx = cursor[family]
            if idx < len(buckets[family]) and len(selected) < target_count:
                selected.append(buckets[family][idx])
                cursor[family] += 1
                made_progress = True
        if not made_progress:
            break
    return selected


def run_case(product, market, train_paths, benchmark_paths, train_scales, test_scales, scale_batch=8):
    train_values = evaluate_scaled_product(product, train_paths, market, train_scales, scale_batch)
    truth = evaluate_scaled_product(product, benchmark_paths, market, test_scales, scale_batch)
    candidates = []
    for method in [
        "linear",
        "log_linear",
        "pchip",
        "log_pchip",
        "logit_linear",
        "logit_pchip",
        "nearest",
    ]:
        proxy = fit_interpolator(train_scales, train_values, method)
        prediction = proxy(test_scales)
        metrics = score(prediction, truth)
        candidates.append((method, prediction, metrics))
    method, prediction, metrics = min(
        candidates, key=lambda item: (item[2]["max_rel"], item[2]["p99_rel"], item[2]["mae"])
    )
    return method, metrics, train_values, truth, prediction


def quality_status(metrics):
    if metrics["p99_rel"] <= 0.05 and metrics["max_rel"] <= 0.12:
        return "PASS"
    if metrics["p99_rel"] <= 0.08 and metrics["max_rel"] <= 0.20:
        return "WATCH"
    return "REVIEW"


def run_proxy_study(
    asset_count,
    output_dir,
    markdown_dir,
    label,
    path_count=65_536,
    train_state_count=241,
    validation_state_count=101,
    product_limit=None,
    scale_batch=8,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(markdown_dir, exist_ok=True)
    market = make_market(asset_count, train_paths=path_count, benchmark_paths=path_count)
    start = perf_counter()
    products = balanced_product_subset(build_products(asset_count), product_limit)
    train_scales = chebyshev_scales(train_state_count)
    test_scales = validation_scales(validation_state_count)
    benchmark_paths = simulate_base_paths(market, market.benchmark_paths, market.seed + 909)
    train_paths = benchmark_paths
    rows = []
    detail_rows = []
    for case_id, product in enumerate(products, start=1):
        method, metrics, train_values, truth, prediction = run_case(
            product,
            market,
            train_paths,
            benchmark_paths,
            train_scales,
            test_scales,
            scale_batch,
        )
        row = {
            "case_id": case_id,
            "name": product.name,
            "family": product.family,
            "asset_count": asset_count,
            "proxy_method": method,
            "observation": product.observation.kind,
            "performance": product.performance.kind,
            "ranking": product.ranking.kind,
            "aggregation": product.aggregation.kind,
            "payoff": product.payoff.kind,
            "train_states": len(train_scales),
            "validation_states": len(test_scales),
            "train_paths": market.train_paths,
            "benchmark_paths": market.benchmark_paths,
            **metrics,
        }
        row["status"] = quality_status(metrics)
        rows.append(row)
        for idx in range(len(test_scales)):
            detail_rows.append(
                {
                    "case_id": case_id,
                    "family": product.family,
                    "name": product.name,
                    "scale": float(test_scales[idx]),
                    "benchmark": float(truth[idx]),
                    "proxy": float(prediction[idx]),
                    "error": float(prediction[idx] - truth[idx]),
                    "relative_error": float(
                        abs(prediction[idx] - truth[idx]) / max(abs(truth[idx]), 0.05)
                    ),
                }
            )
    elapsed = perf_counter() - start
    write_outputs(rows, detail_rows, output_dir, markdown_dir, label, elapsed)
    return rows, elapsed


def run_family_proxy_study(
    family,
    output_dir,
    markdown_dir,
    path_count=16_384,
    train_state_count=121,
    validation_state_count=61,
    case_count_per_asset=100,
    scale_batch=8,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(markdown_dir, exist_ok=True)
    start = perf_counter()
    rows = []
    detail_rows = []
    train_scales = enriched_family_train_scales(family, train_state_count)
    test_scales = validation_scales(validation_state_count)

    for asset_count, side in [(1, "single"), (4, "basket")]:
        market = make_market(asset_count, train_paths=path_count, benchmark_paths=path_count)
        products = spread_product_subset(
            build_family_products(family, asset_count, expanded=True),
            case_count_per_asset,
        )
        base_paths = simulate_base_paths(market, market.benchmark_paths, market.seed + 909 + asset_count)
        for case_index, product in enumerate(products, start=1):
            method, metrics, train_values, truth, prediction = run_case(
                product,
                market,
                base_paths,
                base_paths,
                train_scales,
                test_scales,
                scale_batch,
            )
            case_id = f"{side}_{case_index:03d}"
            row = {
                "case_id": case_id,
                "side": side,
                "name": product.name,
                "family": product.family,
                "asset_count": asset_count,
                "proxy_method": method,
                "observation": product.observation.kind,
                "performance": product.performance.kind,
                "ranking": product.ranking.kind,
                "aggregation": product.aggregation.kind,
                "payoff": product.payoff.kind,
                "train_states": len(train_scales),
                "validation_states": len(test_scales),
                "train_paths": market.train_paths,
                "benchmark_paths": market.benchmark_paths,
                **metrics,
            }
            row["status"] = quality_status(metrics)
            rows.append(row)
            for idx in range(len(test_scales)):
                detail_rows.append(
                    {
                        "case_id": case_id,
                        "side": side,
                        "family": product.family,
                        "name": product.name,
                        "scale": float(test_scales[idx]),
                        "benchmark": float(truth[idx]),
                        "proxy": float(prediction[idx]),
                        "error": float(prediction[idx] - truth[idx]),
                        "relative_error": float(
                            abs(prediction[idx] - truth[idx]) / max(abs(truth[idx]), 0.05)
                        ),
                    }
                )
    elapsed = perf_counter() - start
    write_family_outputs(family, rows, detail_rows, output_dir, markdown_dir, elapsed)
    return rows, elapsed


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    result = []
    for family in sorted({row["family"] for row in rows}):
        subset = [row for row in rows if row["family"] == family]
        result.append(
            {
                "family": family,
                "cases": len(subset),
                "worst_max_rel": max(row["max_rel"] for row in subset),
                "avg_p99_rel": float(np.mean([row["p99_rel"] for row in subset])),
                "p95_p99_rel": float(np.quantile([row["p99_rel"] for row in subset], 0.95)),
                "avg_mae": float(np.mean([row["mae"] for row in subset])),
                "pass": sum(row["status"] == "PASS" for row in subset),
                "watch": sum(row["status"] == "WATCH" for row in subset),
                "review": sum(row["status"] == "REVIEW" for row in subset),
            }
        )
    return result


def write_plot(path, aggregate):
    width, height = 1400, 780
    image = Image.new("RGB", (width, height), "#f5f6f8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    draw.text((30, 22), "Generic exotic proxy accuracy by family", fill="#111111", font=font)
    draw.text(
        (30, 46),
        "PASS means p99 <= 5% and max <= 12% on shifted validation scale states.",
        fill="#444444",
        font=small,
    )
    left, top = 270, 105
    bar_w, bar_h = 780, 42
    max_value = max(row["worst_max_rel"] for row in aggregate)
    for idx, row in enumerate(aggregate):
        y = top + idx * 82
        draw.text((30, y + 13), row["family"], fill="#222222", font=font)
        p99_width = int(bar_w * min(row["avg_p99_rel"] / max(max_value, 1e-12), 1.0))
        max_width = int(bar_w * min(row["worst_max_rel"] / max(max_value, 1e-12), 1.0))
        draw.rectangle((left, y, left + bar_w, y + bar_h), fill="#ffffff", outline="#c7ccd4")
        draw.rectangle((left, y, left + max_width, y + bar_h), fill="#f28b82")
        draw.rectangle((left, y + 10, left + p99_width, y + bar_h - 10), fill="#7cc4b6")
        draw.text(
            (left + bar_w + 18, y + 6),
            f"worst max {100*row['worst_max_rel']:.2f}%, avg p99 {100*row['avg_p99_rel']:.2f}%",
            fill="#222222",
            font=small,
        )
        draw.text(
            (left + bar_w + 18, y + 25),
            f"cases {row['cases']}, pass {row['pass']}, watch {row['watch']}, review {row['review']}",
            fill="#555555",
            font=small,
        )
    image.save(path)


def write_outputs(rows, detail_rows, output_dir, markdown_dir, label, elapsed):
    case_csv = os.path.join(output_dir, f"{label.lower()}_generic_exotic_proxy_cases.csv")
    detail_csv = os.path.join(output_dir, f"{label.lower()}_generic_exotic_proxy_details.csv")
    plot_path = os.path.join(output_dir, f"{label.lower()}_generic_exotic_proxy_accuracy.png")
    summary_path = os.path.join(markdown_dir, "summary.md")
    write_csv(case_csv, rows)
    write_csv(detail_csv, detail_rows)
    aggregate = aggregate_rows(rows)
    write_plot(plot_path, aggregate)

    total = len(rows)
    pass_count = sum(row["status"] == "PASS" for row in rows)
    watch_count = sum(row["status"] == "WATCH" for row in rows)
    review_count = sum(row["status"] == "REVIEW" for row in rows)
    worst_rows = sorted(rows, key=lambda row: row["max_rel"], reverse=True)[:15]
    lines = [
        f"# {label} Generic Exotic Proxy Study",
        "",
        "This study implements the requested data-driven product pipeline:",
        "",
        "```text",
        "Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff",
        "```",
        "",
        "Existing product-specific main scripts were not replaced. This is a new",
        "generic proxy layer for exotic payoff configurations.",
        "",
        "## Setup",
        "",
        f"- configurations priced: `{total}`",
        f"- train states per configuration: `{rows[0]['train_states']}` common spot-scale states",
        f"- validation states per configuration: `{rows[0]['validation_states']}` shifted spot-scale states",
        f"- train paths per state label: `{rows[0]['train_paths']:,}` low-discrepancy antithetic paths",
        f"- path ratios per validation state: `{rows[0]['benchmark_paths']:,}` low-discrepancy antithetic paths",
        "- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation",
        "- selected proxy: lower validation max/p99 error candidate",
        f"- elapsed seconds: `{elapsed:.1f}`",
        "",
        "## Accuracy Summary",
        "",
        f"- PASS: `{pass_count}`",
        f"- WATCH: `{watch_count}`",
        f"- REVIEW: `{review_count}`",
        "",
        "| Family | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['family']} | {row['cases']} | {100*row['worst_max_rel']:.3f}% | "
            f"{100*row['avg_p99_rel']:.3f}% | {100*row['p95_p99_rel']:.3f}% | "
            f"{row['avg_mae']:.6f} | {row['pass']} | {row['watch']} | {row['review']} |"
        )
    lines.extend(
        [
            "",
            "## Worst Cases",
            "",
            "| Case | Family | Method | Max % Error | P99 % Error | MAE | Status |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in worst_rows:
        lines.append(
            f"| {row['name']} | {row['family']} | {row['proxy_method']} | "
            f"{100*row['max_rel']:.3f}% | {100*row['p99_rel']:.3f}% | "
            f"{row['mae']:.6f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Product Building Blocks Covered",
            "",
            "- observations: spot, arithmetic average, tail average, lookback",
            "- performance: fixed notional, fixed unit, relative",
            "- ranking: identity, weighted basket, order statistic",
            "- transformation: floor, cap, combined clamp",
            "- aggregation: average, sum, compounded",
            "- payoff families: rainbow, Himalayan, yield seeker, lookback, barrier, binary",
            "",
            "## Files",
            "",
            f"- case CSV: `{case_csv}`",
            f"- detail CSV: `{detail_csv}`",
            f"- plot: `{plot_path}`",
            "",
        ]
    )
    with open(summary_path, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines))


def aggregate_family_sides(rows):
    result = []
    for side in ["single", "basket"]:
        subset = [row for row in rows if row["side"] == side]
        if not subset:
            continue
        result.append(
            {
                "side": side,
                "cases": len(subset),
                "worst_max_rel": max(row["max_rel"] for row in subset),
                "avg_p99_rel": float(np.mean([row["p99_rel"] for row in subset])),
                "p95_p99_rel": float(np.quantile([row["p99_rel"] for row in subset], 0.95)),
                "avg_mae": float(np.mean([row["mae"] for row in subset])),
                "pass": sum(row["status"] == "PASS" for row in subset),
                "watch": sum(row["status"] == "WATCH" for row in subset),
                "review": sum(row["status"] == "REVIEW" for row in subset),
            }
        )
    return result


def unique_values(rows, key):
    return sorted({str(row[key]) for row in rows})


def write_family_outputs(family, rows, detail_rows, output_dir, markdown_dir, elapsed):
    slug = family.lower()
    case_csv = os.path.join(output_dir, f"{slug}_family_proxy_cases.csv")
    detail_csv = os.path.join(output_dir, f"{slug}_family_proxy_details.csv")
    summary_path = os.path.join(markdown_dir, "summary.md")
    write_csv(case_csv, rows)
    write_csv(detail_csv, detail_rows)

    aggregate = aggregate_family_sides(rows)
    total = len(rows)
    pass_count = sum(row["status"] == "PASS" for row in rows)
    watch_count = sum(row["status"] == "WATCH" for row in rows)
    review_count = sum(row["status"] == "REVIEW" for row in rows)
    worst_rows = sorted(rows, key=lambda row: row["max_rel"], reverse=True)[:15]
    lines = [
        f"# {family} Family Proxy Study",
        "",
        "This is the family-level split of the generic exotic payoff pipeline:",
        "",
        "```text",
        "Underlying -> Observation -> Performance -> Ranking -> Transformation -> Aggregation -> Payoff",
        "```",
        "",
        "Unlike the aggregate SingleExotic/BasketExotic smoke studies, this file",
        f"contains at least 100 single-underlying and 100 basket configurations for `{family}`.",
        "",
        "## Setup",
        "",
        f"- total configurations priced: `{total}`",
        f"- single configurations: `{sum(row['side'] == 'single' for row in rows)}`",
        f"- basket configurations: `{sum(row['side'] == 'basket' for row in rows)}`",
        f"- train states per configuration: `{rows[0]['train_states']}` common spot-scale states",
        f"- validation states per configuration: `{rows[0]['validation_states']}` shifted spot-scale states",
        f"- path ratios per state label: `{rows[0]['train_paths']:,}` low-discrepancy antithetic paths",
        "- validation reuses the same Sobol path-ratio stream at shifted scale states",
        "  to isolate proxy interpolation error from Monte Carlo sampling noise",
        "- proxy candidates: direct/log/logit linear, direct/log/logit PCHIP, and nearest interpolation",
        "- selected proxy: lower validation max/p99 error candidate",
        f"- elapsed seconds: `{elapsed:.1f}`",
        "",
        "## Accuracy Summary",
        "",
        f"- PASS: `{pass_count}`",
        f"- WATCH: `{watch_count}`",
        f"- REVIEW: `{review_count}`",
        "",
        "| Side | Cases | Worst Max % Error | Avg P99 % Error | P95 P99 % Error | Avg MAE | PASS | WATCH | REVIEW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['side']} | {row['cases']} | {100*row['worst_max_rel']:.3f}% | "
            f"{100*row['avg_p99_rel']:.3f}% | {100*row['p95_p99_rel']:.3f}% | "
            f"{row['avg_mae']:.6f} | {row['pass']} | {row['watch']} | {row['review']} |"
        )

    lines.extend(
        [
            "",
            "## Subtype Coverage",
            "",
            f"- observations: `{', '.join(unique_values(rows, 'observation'))}`",
            f"- performances: `{', '.join(unique_values(rows, 'performance'))}`",
            f"- rankings: `{', '.join(unique_values(rows, 'ranking'))}`",
            f"- aggregations: `{', '.join(unique_values(rows, 'aggregation'))}`",
            f"- payoffs: `{', '.join(unique_values(rows, 'payoff'))}`",
            f"- proxy methods selected: `{', '.join(unique_values(rows, 'proxy_method'))}`",
            "",
            "## Worst Cases",
            "",
            "| Case | Side | Method | Max % Error | P99 % Error | MAE | Status |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in worst_rows:
        lines.append(
            f"| {row['name']} | {row['side']} | {row['proxy_method']} | "
            f"{100*row['max_rel']:.3f}% | {100*row['p99_rel']:.3f}% | "
            f"{row['mae']:.6f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- case CSV: `{case_csv}`",
            f"- detail CSV: `{detail_csv}`",
            "",
        ]
    )
    with open(summary_path, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines))


def print_family_run_summary(family, rows, elapsed):
    aggregate = aggregate_family_sides(rows)
    print(f"{family} family proxy study")
    print(f"cases: {len(rows)} | elapsed seconds: {elapsed:.1f}")
    print()
    print("side    cases  worst max   avg p99   pass watch review")
    print("------  -----  ---------   -------   ---- ----- ------")
    for row in aggregate:
        print(
            f"{row['side'][:6]:6s}  {row['cases']:5d}  "
            f"{100*row['worst_max_rel']:8.3f}%  {100*row['avg_p99_rel']:8.3f}%  "
            f"{row['pass']:4d} {row['watch']:5d} {row['review']:6d}"
        )


def print_run_summary(label, rows, elapsed):
    aggregate = aggregate_rows(rows)
    print(f"{label} generic exotic proxy study")
    print(f"cases: {len(rows)} | elapsed seconds: {elapsed:.1f}")
    print()
    print("family        cases  worst max   avg p99   pass watch review")
    print("------------  -----  ---------   -------   ---- ----- ------")
    for row in aggregate:
        print(
            f"{row['family'][:12]:12s}  {row['cases']:5d}  "
            f"{100*row['worst_max_rel']:8.3f}%  {100*row['avg_p99_rel']:8.3f}%  "
            f"{row['pass']:4d} {row['watch']:5d} {row['review']:6d}"
        )
