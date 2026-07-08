from dataclasses import dataclass
from math import erf, exp, log, sqrt

import numpy as np


@dataclass(frozen=True)
class GBMParams:
    s0: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    div_yield: float = 0.02
    vol: float = 0.20
    maturity: float = 1.0
    n_steps: int = 5
    n_paths: int = 100_000
    seed: int = 7
    option_type: str = "call"
    basis_degree: int = 9
    ridge_lambda: float = 1e-6


def simulate_gbm_paths(params: GBMParams) -> tuple[np.ndarray, np.ndarray]:
    """Exact GBM simulation under the risk-neutral measure."""
    rng = np.random.default_rng(params.seed)
    dt = params.maturity / params.n_steps
    times = np.linspace(0.0, params.maturity, params.n_steps + 1)
    paths = np.empty((params.n_paths, params.n_steps + 1))
    paths[:, 0] = params.s0

    drift = (params.rate - params.div_yield - 0.5 * params.vol**2) * dt
    diffusion = params.vol * sqrt(dt)

    z = rng.standard_normal((params.n_paths, params.n_steps))
    for step in range(params.n_steps):
        paths[:, step + 1] = paths[:, step] * np.exp(drift + diffusion * z[:, step])

    return times, paths


def european_payoff(spot: np.ndarray, params: GBMParams) -> np.ndarray:
    if params.option_type == "call":
        return np.maximum(spot - params.strike, 0.0)
    if params.option_type == "put":
        return np.maximum(params.strike - spot, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def polynomial_basis(spot: np.ndarray, params: GBMParams) -> np.ndarray:
    # Centering by s0 keeps the regression numerically tame.
    x = np.asarray(spot, dtype=float) / params.s0 - 1.0
    columns = [x**power for power in range(params.basis_degree + 1)]
    columns.append(european_payoff(np.asarray(spot, dtype=float), params) / params.s0)
    return np.column_stack(columns)


def ridge_coefficients_from_moments(
    xtx: np.ndarray,
    xty: np.ndarray,
    ridge_lambda: float,
    unpenalized_columns: int = 1,
) -> np.ndarray:
    penalty = np.eye(xtx.shape[0]) * ridge_lambda
    penalty[:unpenalized_columns, :unpenalized_columns] = 0.0
    return np.linalg.solve(xtx + penalty, xty)


def fit_ridge_coefficients(
    basis: np.ndarray,
    target: np.ndarray,
    params: GBMParams,
) -> np.ndarray:
    return ridge_coefficients_from_moments(
        basis.T @ basis,
        basis.T @ target,
        params.ridge_lambda,
    )


def asymptotic_anchor_value(
    spot: np.ndarray, tau: float, params: GBMParams
) -> np.ndarray:
    """European option tail asymptote: zero OTM, linear deeply ITM."""
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return european_payoff(spot, params)

    discounted_spot = spot * exp(-params.div_yield * tau)
    discounted_strike = params.strike * exp(-params.rate * tau)
    if params.option_type == "call":
        return np.maximum(discounted_spot - discounted_strike, 0.0)
    if params.option_type == "put":
        return np.maximum(discounted_strike - discounted_spot, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def option_upper_bound(spot: np.ndarray, tau: float, params: GBMParams) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    if params.option_type == "call":
        return spot * exp(-params.div_yield * max(tau, 0.0))
    if params.option_type == "put":
        return np.full_like(spot, params.strike * exp(-params.rate * max(tau, 0.0)))
    raise ValueError("option_type must be 'call' or 'put'")


def asymptotic_window(spot: np.ndarray, tau: float, params: GBMParams) -> np.ndarray:
    """Central window for time value; decays toward deep ITM/OTM tails."""
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return np.zeros_like(spot)

    log_moneyness = np.log(np.maximum(spot, 1e-12) / params.strike)
    width = max(2.0 * params.vol * sqrt(tau), 0.18)
    return np.exp(-0.5 * (log_moneyness / width) ** 2)


def asymptotic_blend_weight(
    spot: np.ndarray, tau: float, params: GBMParams
) -> np.ndarray:
    """Weight on the learned proxy; the complement uses the tail asymptote."""
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return np.zeros_like(spot)

    log_moneyness = np.abs(np.log(np.maximum(spot, 1e-12) / params.strike))
    width = max(2.2 * params.vol * sqrt(tau), 0.22)
    return np.exp(-0.5 * (log_moneyness / width) ** 4)


def normalized_delta_for_cutoff(
    spot: np.ndarray, tau: float, params: GBMParams
) -> np.ndarray:
    """Delta-like moneyness score in [0, 1] for identifying far tails."""
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        if params.option_type == "call":
            return (spot > params.strike).astype(float)
        if params.option_type == "put":
            return (spot < params.strike).astype(float)
        raise ValueError("option_type must be 'call' or 'put'")

    sigma_sqrt_tau = params.vol * sqrt(tau)
    d1 = (
        np.log(spot / params.strike)
        + (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    ) / sigma_sqrt_tau
    call_score = normal_cdf(d1)
    if params.option_type == "call":
        return call_score
    if params.option_type == "put":
        return 1.0 - call_score
    raise ValueError("option_type must be 'call' or 'put'")


def asymptotic_residual_basis(
    spot: np.ndarray, tau: float, params: GBMParams
) -> np.ndarray:
    return polynomial_basis(spot, params) * asymptotic_window(spot, tau, params)[:, None]


def fit_asymptotic_lsmc_proxies(
    times: np.ndarray, paths: np.ndarray, params: GBMParams
) -> list[np.ndarray]:
    terminal_payoff = european_payoff(paths[:, -1], params)
    n_basis = polynomial_basis(np.array([params.s0]), params).shape[1]
    coeffs_by_time = []

    for step, time in enumerate(times):
        tau = params.maturity - time
        target = exp(-params.rate * tau) * terminal_payoff

        if step == params.n_steps:
            coeffs = np.zeros(n_basis)
        elif step == 0:
            coeffs = np.zeros(n_basis)
            basis_at_start = asymptotic_residual_basis(
                np.array([params.s0]), tau, params
            )[0, 0]
            anchor_at_start = asymptotic_anchor_value(
                np.array([params.s0]), tau, params
            )[0]
            coeffs[0] = (target.mean() - anchor_at_start) / basis_at_start
        else:
            spot = paths[:, step]
            anchor = asymptotic_anchor_value(spot, tau, params)
            basis = asymptotic_residual_basis(spot, tau, params)
            coeffs = fit_ridge_coefficients(basis, target - anchor, params)

        coeffs_by_time.append(coeffs)

    return coeffs_by_time


def asymptotic_proxy_value(
    spot: np.ndarray, tau: float, coeffs: np.ndarray, params: GBMParams
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    anchor = asymptotic_anchor_value(spot, tau, params)
    residual = asymptotic_residual_basis(spot, tau, params) @ coeffs
    raw_value = anchor + residual
    return np.minimum(np.maximum(raw_value, anchor), option_upper_bound(spot, tau, params))


def asymptotic_blended_proxy_value(
    spot: np.ndarray, tau: float, coeffs: np.ndarray, params: GBMParams
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    anchor = asymptotic_anchor_value(spot, tau, params)
    learned_value = proxy_value(spot, coeffs, params)
    weight = asymptotic_blend_weight(spot, tau, params)
    raw_value = weight * learned_value + (1.0 - weight) * anchor
    return np.minimum(np.maximum(raw_value, anchor), option_upper_bound(spot, tau, params))


def asymptotic_cutoff_proxy_value(
    spot: np.ndarray,
    tau: float,
    coeffs: np.ndarray,
    params: GBMParams,
    low_delta: float = 0.001,
    high_delta: float = 0.999,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    learned_value = proxy_value(spot, coeffs, params)
    anchor = asymptotic_anchor_value(spot, tau, params)
    delta_score = normalized_delta_for_cutoff(spot, tau, params)
    use_anchor = (delta_score < low_delta) | (delta_score > high_delta)
    return np.where(use_anchor, anchor, learned_value)


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def asymptotic_three_region_weight(
    delta_score: np.ndarray,
    tail_delta: float = 0.001,
    trained_delta: float = 0.01,
) -> np.ndarray:
    """Weight on trained proxy: 0 in tails, 1 centrally, smooth in between."""
    delta_score = np.asarray(delta_score, dtype=float)
    weight = np.ones_like(delta_score)

    low_tail = delta_score <= tail_delta
    low_mix = (delta_score > tail_delta) & (delta_score < trained_delta)
    high_mix = (delta_score > 1.0 - trained_delta) & (
        delta_score < 1.0 - tail_delta
    )
    high_tail = delta_score >= 1.0 - tail_delta

    weight[low_tail | high_tail] = 0.0
    weight[low_mix] = smoothstep(
        (delta_score[low_mix] - tail_delta) / (trained_delta - tail_delta)
    )
    weight[high_mix] = smoothstep(
        ((1.0 - tail_delta) - delta_score[high_mix])
        / (trained_delta - tail_delta)
    )
    return weight


def asymptotic_three_region_proxy_value(
    spot: np.ndarray,
    tau: float,
    coeffs: np.ndarray,
    params: GBMParams,
    tail_delta: float = 0.001,
    trained_delta: float = 0.01,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    learned_value = proxy_value(spot, coeffs, params)
    anchor = asymptotic_anchor_value(spot, tau, params)
    delta_score = normalized_delta_for_cutoff(spot, tau, params)
    weight = asymptotic_three_region_weight(delta_score, tail_delta, trained_delta)
    return weight * learned_value + (1.0 - weight) * anchor


def default_proxy_value(
    spot: np.ndarray, tau: float, coeffs: np.ndarray, params: GBMParams
) -> np.ndarray:
    return asymptotic_three_region_proxy_value(spot, tau, coeffs, params)


def proxy_model_description(params: GBMParams) -> str:
    return (
        f"three-region asymptotic ridge polynomial degree {params.basis_degree} "
        f"plus payoff basis, lambda={params.ridge_lambda:g}"
    )


def fit_lsmc_proxies(
    times: np.ndarray, paths: np.ndarray, params: GBMParams
) -> list[np.ndarray]:
    terminal_payoff = european_payoff(paths[:, -1], params)
    n_basis = polynomial_basis(np.array([params.s0]), params).shape[1]
    coeffs_by_time = []

    for step, time in enumerate(times):
        target = exp(-params.rate * (params.maturity - time)) * terminal_payoff

        if step == 0:
            coeffs = np.zeros(n_basis)
            coeffs[0] = target.mean()
        elif step == params.n_steps:
            coeffs = np.zeros(n_basis)
            coeffs[-1] = params.s0
        else:
            x = polynomial_basis(paths[:, step], params)
            coeffs = fit_ridge_coefficients(x, target, params)

        coeffs_by_time.append(coeffs)

    return coeffs_by_time


def proxy_value(spot: np.ndarray, coeffs: np.ndarray, params: GBMParams) -> np.ndarray:
    return polynomial_basis(spot, params) @ coeffs


def normal_cdf(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def black_scholes_value(spot: np.ndarray, tau: float, params: GBMParams) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    if tau <= 0.0:
        return european_payoff(spot, params)

    sigma_sqrt_tau = params.vol * sqrt(tau)
    d1 = (
        np.log(spot / params.strike)
        + (params.rate - params.div_yield + 0.5 * params.vol**2) * tau
    ) / sigma_sqrt_tau
    d2 = d1 - sigma_sqrt_tau

    if params.option_type == "call":
        return (
            spot * exp(-params.div_yield * tau) * normal_cdf(d1)
            - params.strike * exp(-params.rate * tau) * normal_cdf(d2)
        )

    return (
        params.strike * exp(-params.rate * tau) * normal_cdf(-d2)
        - spot * exp(-params.div_yield * tau) * normal_cdf(-d1)
    )


def summarize_fit(times: np.ndarray, paths: np.ndarray, coeffs_by_time: list[np.ndarray], params: GBMParams) -> None:
    print("Stage 1: European option LSMC proxy")
    print(
        f"GBM: S0={params.s0:.2f}, K={params.strike:.2f}, r={params.rate:.2%}, "
        f"q={params.div_yield:.2%}, vol={params.vol:.2%}, T={params.maturity:.2f}"
    )
    print(
        f"Simulation: {params.n_paths:,} paths, {params.n_steps} time steps, "
        f"{proxy_model_description(params)}"
    )
    print()
    print("time    tau     mean proxy   BS at E[S_t]   grid MAE   grid RMSE")
    print("----    ---     ----------   ----------     --------   ---------")

    for step, time in enumerate(times):
        tau = params.maturity - time
        spots_t = paths[:, step]
        coeffs = coeffs_by_time[step]

        if step == 0:
            grid = np.array([params.s0])
        else:
            lo, hi = np.quantile(spots_t, [0.05, 0.95])
            grid = np.linspace(lo, hi, 31)

        proxy_grid = default_proxy_value(grid, tau, coeffs, params)
        bs_grid = black_scholes_value(grid, tau, params)
        error = proxy_grid - bs_grid

        mean_proxy = default_proxy_value(spots_t, tau, coeffs, params).mean()
        bs_at_mean_spot = black_scholes_value(np.array([spots_t.mean()]), tau, params)[0]

        print(
            f"{time:4.2f}   {tau:4.2f}    {mean_proxy:10.4f}   {bs_at_mean_spot:10.4f}   "
            f"{np.mean(np.abs(error)):8.4f}   {np.sqrt(np.mean(error**2)):9.4f}"
        )


def main() -> None:
    params = GBMParams()
    times, paths = simulate_gbm_paths(params)
    coeffs_by_time = fit_lsmc_proxies(times, paths, params)
    summarize_fit(times, paths, coeffs_by_time, params)

    print()
    print("Example proxy coefficients by time:")
    for time, coeffs in zip(times, coeffs_by_time):
        rounded = ", ".join(f"{value:.6f}" for value in coeffs)
        print(f"t={time:.2f}: [{rounded}]")


if __name__ == "__main__":
    main()
