import csv
import os
from dataclasses import dataclass
from math import erf, exp, log, sqrt
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "tmp", "single_feature_term_structure")
CSV_PATH = os.path.join(OUTPUT_DIR, "single_feature_term_structure_sensitivity.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "single_feature_term_structure_sensitivity.png")
SUMMARY_PATH = os.path.join(
    ROOT,
    "Markdown",
    "MethodStudy",
    "results",
    "term_structure_single_feature_summary.md",
)


@dataclass(frozen=True)
class Config:
    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    div_yield: float = 0.015
    paths: int = 262_144
    seed: int = 20260712
    relative_floor: float = 0.05
    barrier: float = 75.0
    notional: float = 100.0
    autocall_barrier: float = 1.00
    coupon_barrier: float = 0.75
    protection_barrier: float = 0.65
    coupon_per_obs: float = 0.025


SEGMENTS = 4
DT = 1.0 / SEGMENTS
RATE_FLAT = np.array([0.04, 0.04, 0.04, 0.04])
RATE_FRONT = np.array([0.075, 0.055, 0.025, 0.005])
RATE_BACK = RATE_FRONT[::-1]
VOL_FLAT = np.full(SEGMENTS, sqrt(np.mean(np.array([0.40, 0.30, 0.15, 0.10]) ** 2)))
VOL_FRONT = np.array([0.40, 0.30, 0.15, 0.10])
VOL_BACK = VOL_FRONT[::-1]


SCENARIO_PAIRS = [
    ("rate shape only", RATE_FRONT, VOL_FLAT, RATE_BACK, VOL_FLAT),
    ("vol shape only", RATE_FLAT, VOL_FRONT, RATE_FLAT, VOL_BACK),
    ("rate+vol shape", RATE_FRONT, VOL_FRONT, RATE_BACK, VOL_BACK),
]


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def total_rate(rates):
    return float(np.sum(rates) * DT)


def total_variance(vols):
    return float(np.sum(vols * vols) * DT)


def effective_vol(vols):
    return sqrt(total_variance(vols))


def generalized_bs_call(rates, vols, cfg):
    r_int = total_rate(rates)
    q_int = cfg.div_yield * cfg.maturity
    v_int = total_variance(vols)
    vol_sqrt = sqrt(v_int)
    d1 = (log(cfg.spot / cfg.strike) + r_int - q_int + 0.5 * v_int) / vol_sqrt
    d2 = d1 - vol_sqrt
    return (
        cfg.spot * exp(-q_int) * normal_cdf(d1)
        - cfg.strike * exp(-r_int) * normal_cdf(d2)
    )


def antithetic_normals(cfg):
    half = cfg.paths // 2
    rng = np.random.default_rng(cfg.seed)
    base = rng.standard_normal((half, SEGMENTS))
    return np.vstack([base, -base])


def simulate_paths(rates, vols, normals, cfg):
    spots = np.empty((normals.shape[0], SEGMENTS))
    current = np.full(normals.shape[0], cfg.spot)
    for step in range(SEGMENTS):
        drift = (rates[step] - cfg.div_yield - 0.5 * vols[step] ** 2) * DT
        diffusion = vols[step] * sqrt(DT)
        current = current * np.exp(drift + diffusion * normals[:, step])
        spots[:, step] = current
    return spots


def terminal_random_payoff(spot):
    strikes = np.array([45.0, 70.0, 90.0, 110.0, 135.0, 165.0, 200.0])
    values = np.array([8.0, 15.0, 27.0, 19.0, 54.0, 42.0, 68.0])
    return np.interp(spot, strikes, values, left=values[0], right=values[-1])


def random_terminal_value(rates, vols, normals, cfg):
    r_int = total_rate(rates)
    v_int = total_variance(vols)
    z = np.sum(vols[None, :] * sqrt(DT) * normals, axis=1) / sqrt(v_int)
    terminal = cfg.spot * np.exp(
        r_int - cfg.div_yield * cfg.maturity - 0.5 * v_int + sqrt(v_int) * z
    )
    return exp(-r_int) * float(np.mean(terminal_random_payoff(terminal)))


def asian_arithmetic_call(rates, vols, normals, cfg):
    paths = simulate_paths(rates, vols, normals, cfg)
    avg = np.mean(paths, axis=1)
    payoff = np.maximum(avg - cfg.strike, 0.0)
    return exp(-total_rate(rates)) * float(np.mean(payoff))


def barrier_down_out_call(rates, vols, normals, cfg):
    paths = simulate_paths(rates, vols, normals, cfg)
    log_barrier = log(cfg.barrier)
    previous = np.full(normals.shape[0], log(cfg.spot))
    survival = np.ones(normals.shape[0])
    for step in range(SEGMENTS):
        current = np.log(paths[:, step])
        valid = (previous > log_barrier) & (current > log_barrier)
        segment_survival = np.zeros_like(survival)
        exponent = -2.0 * (previous - log_barrier) * (current - log_barrier) / (
            vols[step] ** 2 * DT
        )
        segment_survival[valid] = 1.0 - np.exp(np.minimum(exponent[valid], 0.0))
        survival *= np.clip(segment_survival, 0.0, 1.0)
        previous = current
    terminal = paths[:, -1]
    payoff = np.maximum(terminal - cfg.strike, 0.0) * survival
    return exp(-total_rate(rates)) * float(np.mean(payoff))


def autocallable_value(rates, vols, normals, cfg):
    paths = simulate_paths(rates, vols, normals, cfg)
    active = np.ones(normals.shape[0], dtype=bool)
    value = np.zeros(normals.shape[0])
    cumulative_rates = np.cumsum(rates) * DT
    for step in range(SEGMENTS):
        obs_number = step + 1
        spot = paths[:, step]
        discount = np.exp(-cumulative_rates[step])
        final = step == SEGMENTS - 1
        if not final:
            called = active & (spot >= cfg.autocall_barrier * cfg.strike)
            value[called] = (
                discount
                * cfg.notional
                * (1.0 + cfg.coupon_per_obs * obs_number)
            )
            active[called] = False
        else:
            coupon_redemption = cfg.notional * (1.0 + cfg.coupon_per_obs * obs_number)
            redemption = np.where(
                spot >= cfg.coupon_barrier * cfg.strike,
                coupon_redemption,
                np.where(
                    spot >= cfg.protection_barrier * cfg.strike,
                    cfg.notional,
                    cfg.notional * spot / cfg.strike,
                ),
            )
            value[active] = discount * redemption[active]
    return float(np.mean(value))


def put_intrinsic(spot, strike):
    return np.maximum(strike - np.asarray(spot), 0.0)


def early_exercise_put_value(rates, vols, exercise_steps, cfg):
    spots = np.exp(np.linspace(log(20.0), log(260.0), 501))
    value = put_intrinsic(spots, cfg.strike)
    nodes, weights = np.polynomial.hermite.hermgauss(15)
    normal_nodes = sqrt(2.0) * nodes
    normal_weights = weights / sqrt(np.pi)

    repeated_rates = np.repeat(rates, exercise_steps // SEGMENTS)
    repeated_vols = np.repeat(vols, exercise_steps // SEGMENTS)
    dt = cfg.maturity / exercise_steps

    for step in range(exercise_steps - 1, -1, -1):
        r = repeated_rates[step]
        vol = repeated_vols[step]
        drift = (r - cfg.div_yield - 0.5 * vol * vol) * dt
        diffusion = vol * sqrt(dt)
        continuation = np.zeros_like(spots)
        for node, weight in zip(normal_nodes, normal_weights):
            next_spot = spots * np.exp(drift + diffusion * node)
            continuation += weight * np.interp(
                next_spot,
                spots,
                value,
                left=value[0],
                right=value[-1],
            )
        continuation *= exp(-r * dt)
        value = np.maximum(put_intrinsic(spots, cfg.strike), continuation)
    return float(np.interp(cfg.spot, spots, value))


def bermudan_put_value(rates, vols, cfg):
    return early_exercise_put_value(rates, vols, SEGMENTS, cfg)


def american_put_value(rates, vols, cfg):
    return early_exercise_put_value(rates, vols, 24, cfg)


PRODUCTS = [
    ("European call", "terminal", generalized_bs_call),
    ("Random terminal payoff", "terminal", random_terminal_value),
    ("Asian arithmetic call", "path_average", asian_arithmetic_call),
    ("Bermudan put", "early_exercise", bermudan_put_value),
    ("American put", "early_exercise", american_put_value),
    ("Barrier down-out call", "barrier", barrier_down_out_call),
    ("Autocallable note", "observation_dates", autocallable_value),
]


def evaluate_product(product, rates, vols, normals, cfg):
    name, _, function = product
    if name in {"European call", "Bermudan put", "American put"}:
        return function(rates, vols, cfg)
    return function(rates, vols, normals, cfg)


def feature_recommendation(product_name):
    if product_name in {"European call", "Random terminal payoff"}:
        return "terminal integrated rate and variance"
    if product_name == "Asian arithmetic call":
        return "event-date forwards, average variance, front/back vol slope"
    if product_name in {"Bermudan put", "American put"}:
        return "time-step discount/drift/variance; remaining curve summaries if curves vary"
    if product_name == "Barrier down-out call":
        return "log barrier distance, local next-segment variance, remaining variance"
    if product_name == "Autocallable note":
        return "discount factors and cumulative variances to observation dates"
    return "product-specific event summaries"


def run_study(cfg):
    normals = antithetic_normals(cfg)
    rows = []
    for product in PRODUCTS:
        product_name, product_type, _ = product
        for pair_name, rates_a, vols_a, rates_b, vols_b in SCENARIO_PAIRS:
            value_a = evaluate_product(product, rates_a, vols_a, normals, cfg)
            value_b = evaluate_product(product, rates_b, vols_b, normals, cfg)
            diff = value_a - value_b
            rel = abs(diff) / max(abs(value_a), abs(value_b), cfg.relative_floor)
            rows.append(
                {
                    "product": product_name,
                    "product_type": product_type,
                    "scenario_pair": pair_name,
                    "front_value": value_a,
                    "back_value": value_b,
                    "signed_difference": diff,
                    "relative_difference": rel,
                    "recommendation": feature_recommendation(product_name),
                }
            )
    return rows


def write_csv(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value):
    return f"{100.0 * value:.3f}%"


def write_summary(rows, elapsed, cfg):
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    worst_by_product = []
    for product in [item[0] for item in PRODUCTS]:
        subset = [row for row in rows if row["product"] == product]
        worst_by_product.append(max(subset, key=lambda row: row["relative_difference"]))

    lines = [
        "# Single-Feature Term-Structure Sensitivity Study",
        "",
        "This study checks whether deterministic rate and volatility curve shape can",
        "be collapsed for the other single-underlying proxy products.",
        "",
        "Each scenario pair has the same total rate integral and the same total",
        "variance integral. Only the timing of rates and volatility changes.",
        "",
        "## Curves",
        "",
        f"- flat rate: `{RATE_FLAT.tolist()}`",
        f"- front-loaded rate: `{RATE_FRONT.tolist()}`",
        f"- back-loaded rate: `{RATE_BACK.tolist()}`",
        f"- flat effective volatility: `{VOL_FLAT.round(6).tolist()}`",
        f"- front-loaded volatility: `{VOL_FRONT.tolist()}`",
        f"- back-loaded volatility: `{VOL_BACK.tolist()}`",
        f"- total rate integral in each pair: `{total_rate(RATE_FRONT):.6f}`",
        f"- total variance integral in each vol-shape pair: `{total_variance(VOL_FRONT):.6f}`",
        f"- MC paths for path products: `{cfg.paths:,}` antithetic paths",
        f"- elapsed seconds: `{elapsed:.1f}`",
        "",
        "## Worst Shape Sensitivity By Product",
        "",
        "| Product | Type | Worst Pair | Front Value | Back Value | Signed Diff | Relative Diff | Feature Recommendation |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in worst_by_product:
        lines.append(
            f"| {row['product']} | {row['product_type']} | {row['scenario_pair']} | "
            f"{row['front_value']:.6f} | {row['back_value']:.6f} | "
            f"{row['signed_difference']:.6f} | {pct(row['relative_difference'])} | "
            f"{row['recommendation']} |"
        )

    lines.extend(
        [
            "",
            "## Full Sensitivity Table",
            "",
            "| Product | Scenario Pair | Front Value | Back Value | Signed Diff | Relative Diff |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['product']} | {row['scenario_pair']} | "
            f"{row['front_value']:.6f} | {row['back_value']:.6f} | "
            f"{row['signed_difference']:.6f} | {pct(row['relative_difference'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Terminal-only payoffs are invariant to curve shape once total rate and",
            "total variance are fixed. That includes European options and random",
            "terminal payoffs.",
            "",
            "Path-dependent products are not invariant. Asian, barrier, early-exercise,",
            "and autocallable values changed when volatility or rates were moved earlier",
            "or later in time. For these products, keeping a one-dimensional spot proxy",
            "is reasonable only when the curve is fixed as part of the model",
            "configuration. If the curve varies across trades or scenarios, add",
            "event-date summaries rather than raw curve knots.",
            "",
            "Practical rule:",
            "",
            "```text",
            "terminal payoff: integrated R and V are enough",
            "scheduled payoff: discount factors and variances to event dates",
            "barrier payoff: local segment variance near the barrier matters",
            "early exercise: step-specific discount/drift/variance in the backward recursion",
            "```",
            "",
            f"CSV: `{CSV_PATH}`",
            f"Plot: `{PLOT_PATH}`",
            "",
        ]
    )
    with open(SUMMARY_PATH, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines))


def draw_plot(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    products = [item[0] for item in PRODUCTS]
    pairs = [item[0] for item in SCENARIO_PAIRS]
    values = np.zeros((len(products), len(pairs)))
    for i, product in enumerate(products):
        for j, pair in enumerate(pairs):
            for row in rows:
                if row["product"] == product and row["scenario_pair"] == pair:
                    values[i, j] = 100.0 * row["relative_difference"]
                    break

    width, height = 1420, 820
    image = Image.new("RGB", (width, height), "#f5f6f8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    draw.text((30, 24), "Term-structure shape sensitivity", fill="#111111", font=font)
    draw.text(
        (30, 48),
        "Each pair has the same total rate and total variance; only timing changes.",
        fill="#444444",
        font=small,
    )

    left, top = 260, 100
    cell_w, cell_h = 250, 72
    max_value = max(float(np.max(values)), 1e-12)
    for j, pair in enumerate(pairs):
        draw.text((left + j * cell_w + 16, top - 28), pair, fill="#222222", font=small)
    for i, product in enumerate(products):
        y = top + i * cell_h
        draw.text((30, y + 24), product, fill="#222222", font=small)
        for j in range(len(pairs)):
            x = left + j * cell_w
            intensity = min(values[i, j] / max_value, 1.0)
            red = int(245 - 60 * (1.0 - intensity))
            green = int(247 - 170 * intensity)
            blue = int(249 - 190 * intensity)
            color = (red, green, blue)
            draw.rectangle((x, y, x + cell_w - 8, y + cell_h - 8), fill=color, outline="#c7ccd4")
            draw.text(
                (x + 18, y + 24),
                f"{values[i, j]:.3f}%",
                fill="#111111",
                font=font,
            )
    draw.text(
        (30, height - 52),
        "Low values mean curve shape can be collapsed; high values mean event timing or local variance matters.",
        fill="#333333",
        font=small,
    )
    image.save(PLOT_PATH)


def main():
    cfg = Config()
    start = perf_counter()
    rows = run_study(cfg)
    elapsed = perf_counter() - start
    write_csv(rows)
    draw_plot(rows)
    write_summary(rows, elapsed, cfg)

    print("Single-feature term-structure sensitivity study")
    print(f"path simulations: {cfg.paths:,} antithetic paths")
    print()
    print("product                  worst pair          relative diff   recommendation")
    print("----------------------   -----------------   -------------   --------------")
    for product in [item[0] for item in PRODUCTS]:
        subset = [row for row in rows if row["product"] == product]
        worst = max(subset, key=lambda row: row["relative_difference"])
        print(
            f"{product[:22]:22s}   {worst['scenario_pair'][:17]:17s}   "
            f"{pct(worst['relative_difference']):>13s}   {worst['recommendation']}"
        )
    print()
    print(f"summary written to: {SUMMARY_PATH}")
    print(f"plot written to: {PLOT_PATH}")
    print(f"csv written to: {CSV_PATH}")
    print(f"elapsed seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
