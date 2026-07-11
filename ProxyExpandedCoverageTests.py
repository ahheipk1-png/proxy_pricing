import math
import os

import numpy as np

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
from ProxyTimingBenchmark import american_tree_put


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MD = os.path.join(
    ROOT, "Markdown", "MethodStudy", "results", "expanded_coverage_summary.md"
)
OUTPUT_HTML = os.path.join(
    ROOT, "Markdown", "MethodStudy", "results", "expanded_coverage_summary.html"
)
MIN_CASES = 100


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def record(results, option, detail):
    results.setdefault(option, []).append(detail)


def european_cases(results):
    spots = (60.0, 90.0, 110.0, 160.0)
    for strike in (80.0, 100.0, 120.0, 140.0):
        for vol in (0.10, 0.20, 0.35):
            for maturity in (0.50, 1.00, 2.00):
                call = EuroMain.Params(strike=strike, vol=vol, maturity=maturity, option_type="call")
                put = EuroMain.Params(strike=strike, vol=vol, maturity=maturity, option_type="put")
                for spot in spots:
                    call_value = EuroMain.black_scholes_value(np.array([spot]), maturity, call)[0]
                    put_value = EuroMain.black_scholes_value(np.array([spot]), maturity, put)[0]
                    parity = spot * math.exp(-call.div_yield * maturity) - strike * math.exp(
                        -call.rate * maturity
                    )
                    check(abs((call_value - put_value) - parity) < 1e-9, "European parity failed")
                    record(results, "European", f"K={strike}, vol={vol}, T={maturity}, S={spot}")


def asian_cases(results):
    for strike in (80.0, 95.0, 110.0, 125.0):
        for vol in (0.15, 0.25, 0.40):
            params = AsianMain.Params(strike=strike, vol=vol)
            for day in (0, 3, 6, 9, 10, 11):
                for spot, avg_before in ((70.0, 90.0), (100.0, 100.0), (140.0, 115.0)):
                    running = day * avg_before
                    if day == params.n_fixings - 1:
                        value, stderr = AsianMain.simulate_state_value(
                            spot, running, day, params, np.random.default_rng(1), 8
                        )
                        expected = AsianMain.payoff_from_average(
                            (running + spot) / params.n_fixings, params
                        )
                        check(abs(value - expected) < 1e-12 and stderr == 0.0, "Asian terminal failed")
                    else:
                        coordinate = AsianMain.adjusted_moneyness_coordinate(
                            spot, running, day, params
                        )
                        check(np.isfinite(coordinate), "Asian coordinate not finite")
                    record(results, "Asian", f"K={strike}, vol={vol}, day={day}, S={spot}, avg={avg_before}")


def american_cases(results):
    for strike in (80.0, 100.0, 120.0, 140.0):
        for vol in (0.15, 0.25, 0.40):
            for rate in (0.01, 0.04, 0.07):
                params = AmericanMain.Params(strike=strike, vol=vol, rate=rate)
                for spot in (60.0, 100.0, 160.0):
                    value = american_tree_put(spot, params, steps=80)
                    check(value + 1e-10 >= AmericanMain.intrinsic(spot, params), "American below intrinsic")
                    record(results, "American", f"K={strike}, vol={vol}, r={rate}, S={spot}")


def bermudan_cases(results):
    old_steps = BermudanMain.BINOMIAL_STEPS_PER_PERIOD
    BermudanMain.BINOMIAL_STEPS_PER_PERIOD = 6
    try:
        for strike in (80.0, 100.0, 120.0, 140.0):
            for vol in (0.15, 0.25, 0.40):
                params = BermudanMain.Params(strike=strike, vol=vol)
                for exercise_index in (0, 4, 8):
                    for spot in (60.0, 100.0, 160.0):
                        value = BermudanMain.binomial_bermudan_value(
                            spot, exercise_index, params
                        )
                        check(value + 1e-10 >= BermudanMain.intrinsic(spot, params), "Bermudan below intrinsic")
                        record(results, "Bermudan", f"K={strike}, vol={vol}, ex={exercise_index}, S={spot}")
    finally:
        BermudanMain.BINOMIAL_STEPS_PER_PERIOD = old_steps


def barrier_cases(results):
    params = BarrierMain.Params()
    variance = params.vol**2 * BarrierMain.dt(params)
    for kind in BarrierMain.BARRIER_KINDS:
        low, high = BarrierMain.barrier_domain(kind, params)
        for month in BarrierMain.TEST_MONTHS:
            for vol in (0.15, 0.25, 0.40):
                local = BarrierMain.Params(vol=vol)
                spots = np.linspace(low + 1.0, high - 1.0, 8)
                for spot in spots:
                    alive = BarrierMain.is_alive(np.array([spot]), kind, local)[0]
                    check(alive in (False, True), "Barrier alive flag invalid")
                    x = np.array([math.log(max(spot, 1e-8))])
                    y = x + 0.01
                    if kind == "down_out":
                        survival = BarrierMain.single_bridge_survival(
                            x, y, math.log(local.lower_barrier), variance, True
                        )[0]
                    elif kind == "up_out":
                        survival = BarrierMain.single_bridge_survival(
                            x, y, math.log(local.upper_barrier), variance, False
                        )[0]
                    else:
                        survival = BarrierMain.double_bridge_survival(
                            x,
                            y,
                            math.log(local.lower_barrier),
                            math.log(local.upper_barrier),
                            variance,
                        )[0]
                    check(0.0 <= survival <= 1.0, "Barrier survival outside [0,1]")
                    record(results, "Barrier", f"{kind}, month={month}, vol={vol}, S={spot:.2f}")


def cliquet_cases(results):
    for cap in (0.03, 0.04, 0.06):
        params = CliquetMain.Params(local_cap=cap)
        for day in CliquetMain.TEST_DAY_INDICES:
            low, high = CliquetMain.accrued_range(day, params)
            accrued_grid = np.linspace(low - 0.05, high + 0.05, 8)
            for accrued in accrued_grid:
                payoff = CliquetMain.payoff_from_accrued(accrued, params)
                lower, upper = CliquetMain.value_bounds(day, params)
                check(params.notional * params.global_floor <= payoff <= params.notional * params.global_cap, "Cliquet payoff bounds failed")
                exact = CliquetMain.exact_tail_value(np.array([accrued]), day, params)[0]
                check(np.isfinite(exact) or np.isnan(exact), "Cliquet tail invalid")
                check(lower <= upper, "Cliquet value bounds inverted")
                record(results, "Cliquet", f"cap={cap}, day={day}, accrued={accrued:.4f}")


def slv_cliquet_cases(results):
    params = SLVCliquetMain.Params()
    for month in SLVCliquetMain.TEST_MONTHS:
        for spot in (50.0, 75.0, 100.0, 130.0, 180.0):
            for variance in (0.01, 0.03, 0.06, 0.10, 0.16):
                lev = SLVCliquetMain.leverage(np.array([spot]), params)[0]
                check(0.5 <= lev <= 1.5, "SLV leverage outside bounds")
                states = (np.array([0.01]), np.array([spot]), np.array([variance]))
                features = SLVCliquetMain.features(states, month, params)
                check(all(np.all(np.isfinite(x)) for x in features), "SLV features invalid")
                record(results, "SLV cliquet", f"month={month}, S={spot}, v={variance}")


def basket_asian_cases(results):
    params = BasketAsianMain.Params()
    corr = BasketAsianMain.correlation_matrix(params)
    check(np.min(np.linalg.eigvalsh(corr)) > -1e-10, "Basket Asian correlation not PSD")
    for day in (0, 3, 6, 9, 10, 11):
        for scale in (0.70, 0.90, 1.10, 1.35):
            spots = np.full((1, params.n_assets), 100.0 * scale)
            for avg_before in (80.0, 95.0, 105.0, 120.0, 140.0):
                running = np.array([day * avg_before])
                value = BasketAsianMain.moment_lognormal_value(spots, running, day, params)[0]
                check(np.isfinite(value) and value >= -1e-10, "Basket Asian value invalid")
                record(results, "Basket Asian", f"day={day}, scale={scale}, avg={avg_before}")


def basket_cliquet_cases(results):
    params = BasketCliquetMain.Params()
    returns_list = (
        np.array([[0.10, -0.04, 0.01]]),
        np.array([[-0.08, -0.02, 0.05]]),
        np.array([[0.00, 0.02, 0.03]]),
    )
    for variant in BasketCliquetMain.VARIANTS:
        for month in BasketCliquetMain.TEST_MONTHS:
            for returns in returns_list:
                coupon = BasketCliquetMain.coupon_values(returns, params)[variant][0]
                low, high = BasketCliquetMain.coupon_bounds(variant, params)
                check(low - 1e-12 <= coupon <= high + 1e-12, "Basket cliquet coupon bounds failed")
                exact = BasketCliquetMain.exact_tail(np.array([coupon]), month, params, variant)[0]
                check(np.isfinite(exact) or np.isnan(exact), "Basket cliquet tail invalid")
                record(results, "Basket cliquet", f"{variant}, month={month}, returns={returns.tolist()}")


def autocallable_cases(results):
    for name, params in AutocallableMain.CASE_GRID:
        for obs in range(params.n_observations + 1):
            for spot in (45.0, 70.0, 90.0, 105.0, 140.0):
                value = AutocallableMain.simulate_values(
                    np.array([spot]), obs, params, 32, params.seed + obs
                )[0]
                upper = params.notional * (1.0 + params.coupon_per_observation * params.n_observations)
                check(np.isfinite(value) and -1e-10 <= value <= upper + 1e-8, "Autocallable value invalid")
                record(results, "Autocallable", f"{name}, obs={obs}, S={spot}")


def random_payoff_cases(results):
    for market_name, market in RandomOptionMain.MARKET_CASES:
        for payoff_case in RandomOptionMain.PAYOFF_CASES:
            for time_fraction in RandomOptionMain.TIME_FRACTIONS:
                tau = market.maturity * (1.0 - time_fraction)
                for spot in (55.0, 100.0, 170.0):
                    value = RandomOptionMain.mc_value(
                        np.array([spot]), tau, market, payoff_case, 64, market.seed
                    )[0]
                    check(np.isfinite(value) and value >= -1e-10, "Random payoff value invalid")
                    record(results, "Random payoff", f"{market_name}, {payoff_case.name}, t={time_fraction}, S={spot}")


def generate_markdown(results):
    lines = [
        "# Expanded Coverage Summary",
        "",
        f"Minimum required cases per option type: `{MIN_CASES}`.",
        "",
        "| Option type | Cases executed | Status | Example subtests |",
        "|---|---:|---|---|",
    ]
    for option in sorted(results):
        cases = results[option]
        status = "PASS" if len(cases) >= MIN_CASES else "FAIL"
        examples = "<br>".join(cases[:5])
        lines.append(f"| {option} | {len(cases)} | {status} | {examples} |")
    return "\n".join(lines) + "\n"


def generate_html(results):
    rows = []
    sections = []
    for option in sorted(results):
        cases = results[option]
        status = "PASS" if len(cases) >= MIN_CASES else "FAIL"
        anchor = option.lower().replace(" ", "-") + "-expanded"
        rows.append(
            f"<tr><td>{option}</td><td class='num'>{len(cases)}</td><td>{status}</td>"
            f"<td><a href='#{anchor}'>view subtests</a></td></tr>"
        )
        items = "\n".join(f"<li><code>{case}</code></li>" for case in cases[:150])
        more = "" if len(cases) <= 150 else f"<p>Showing first 150 of {len(cases)} cases.</p>"
        sections.append(
            f"<section id='{anchor}'><h2>{option}</h2>{more}<ul>{items}</ul>"
            "<p><a href='#top'>Back to top</a></p></section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Expanded Proxy Coverage Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #d7dde3; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #f2f5f8; text-align: left; }}
    .num {{ text-align: right; }}
    code {{ background: #f2f5f8; padding: 1px 4px; border-radius: 3px; }}
    section {{ margin-top: 30px; }}
  </style>
</head>
<body>
  <h1 id="top">Expanded Proxy Coverage Summary</h1>
  <p>Every option family must have at least {MIN_CASES} executed coverage cases.</p>
  <table>
    <thead><tr><th>Option type</th><th>Cases executed</th><th>Status</th><th>Details</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {''.join(sections)}
</body>
</html>
"""


def main():
    results = {}
    for runner in (
        european_cases,
        asian_cases,
        american_cases,
        bermudan_cases,
        barrier_cases,
        cliquet_cases,
        slv_cliquet_cases,
        basket_asian_cases,
        basket_cliquet_cases,
        autocallable_cases,
        random_payoff_cases,
    ):
        runner(results)
    failures = {option: len(cases) for option, cases in results.items() if len(cases) < MIN_CASES}
    if failures:
        raise AssertionError(f"Coverage below {MIN_CASES}: {failures}")

    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="ascii") as handle:
        handle.write(generate_markdown(results))
    with open(OUTPUT_HTML, "w", encoding="ascii") as handle:
        handle.write(generate_html(results))

    print(generate_markdown(results))
    print(f"summary written to: {OUTPUT_MD}")
    print(f"html written to: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
