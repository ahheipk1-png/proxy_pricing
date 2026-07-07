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
    basis_degree: int = 5


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
        else:
            x = polynomial_basis(paths[:, step], params)
            coeffs, *_ = np.linalg.lstsq(x, target, rcond=None)

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
        f"polynomial degree {params.basis_degree} plus payoff basis"
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

        proxy_grid = proxy_value(grid, coeffs, params)
        bs_grid = black_scholes_value(grid, tau, params)
        error = proxy_grid - bs_grid

        mean_proxy = proxy_value(spots_t, coeffs, params).mean()
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
