import math
import unittest
from dataclasses import replace

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


class ProxyCorrectnessTests(unittest.TestCase):
    def assertAllClose(self, actual, expected, atol=1e-10, rtol=1e-10):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    def test_european_put_call_parity_and_d1_inverse(self):
        spots = np.array([60.0, 80.0, 100.0, 125.0, 170.0])
        tau = 1.35
        call = EuroMain.Params(option_type="call", strike=105.0, vol=0.31)
        put = replace(call, option_type="put")

        call_value = EuroMain.black_scholes_value(spots, tau, call)
        put_value = EuroMain.black_scholes_value(spots, tau, put)
        parity = spots * math.exp(-call.div_yield * tau) - call.strike * math.exp(
            -call.rate * tau
        )
        self.assertAllClose(call_value - put_value, parity, atol=1e-9, rtol=1e-9)

        d1 = np.linspace(-4.0, 4.0, 9)
        round_trip = EuroMain.d1_from_spot(EuroMain.spot_from_d1(d1, tau, call), tau, call)
        self.assertAllClose(round_trip, d1, atol=1e-12, rtol=1e-12)

        terminal = EuroMain.black_scholes_value(spots, 0.0, call)
        self.assertAllClose(terminal, EuroMain.payoff(spots, call))

    def test_asian_terminal_and_one_fixing_left_identity(self):
        params = AsianMain.Params()
        day_index = params.n_fixings - 2
        spot = 103.0
        running_sum = 9.0 * 101.0 + 97.0
        m = AsianMain.future_count(day_index, params)
        self.assertEqual(m, 1)

        strike_adj = AsianMain.adjusted_strike(spot, running_sum, day_index, params)
        euro_params = EuroMain.Params(
            strike=float(strike_adj),
            rate=params.rate,
            div_yield=params.div_yield,
            vol=params.vol,
            maturity=params.maturity,
            option_type=params.option_type,
        )
        expected = (1.0 / params.n_fixings) * EuroMain.black_scholes_value(
            np.array([spot]), AsianMain.daily_dt(params), euro_params
        )[0]
        self.assertAlmostEqual(
            AsianMain.geometric_asian_exact(spot, running_sum, day_index, params),
            expected,
            places=10,
        )

        terminal_day = params.n_fixings - 1
        value, stderr = AsianMain.simulate_state_value(
            110.0,
            11.0 * 98.0,
            terminal_day,
            params,
            np.random.default_rng(1),
            16,
        )
        expected_terminal = AsianMain.payoff_from_average((11.0 * 98.0 + 110.0) / 12.0, params)
        self.assertAlmostEqual(value, float(expected_terminal), places=12)
        self.assertEqual(stderr, 0.0)

    def test_american_small_dynamic_programming_is_finite_and_above_intrinsic(self):
        original = {
            "EXERCISE_STEPS": AmericanMain.EXERCISE_STEPS,
            "TRAIN_STATES": AmericanMain.TRAIN_STATES,
            "PATHS_PER_STATE": AmericanMain.PATHS_PER_STATE,
            "TEST_STEPS": AmericanMain.TEST_STEPS,
            "SPOT_MIN": AmericanMain.SPOT_MIN,
            "SPOT_MAX": AmericanMain.SPOT_MAX,
        }
        try:
            AmericanMain.EXERCISE_STEPS = 6
            AmericanMain.TRAIN_STATES = 31
            AmericanMain.PATHS_PER_STATE = 128
            AmericanMain.TEST_STEPS = [0, 3, 6]
            AmericanMain.SPOT_MIN = 45.0
            AmericanMain.SPOT_MAX = 190.0
            params = AmericanMain.Params()
            proxies = AmericanMain.train_proxy(params)
            spots = np.array([70.0, 100.0, 140.0])
            for step in AmericanMain.TEST_STEPS:
                values = proxies[step](spots)
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertTrue(np.all(values + 1e-10 >= AmericanMain.intrinsic(spots, params)))
            self.assertAllClose(proxies[6](spots), AmericanMain.intrinsic(spots, params))
        finally:
            for name, value in original.items():
                setattr(AmericanMain, name, value)

    def test_barrier_survival_bounds_and_in_out_parity(self):
        params = BarrierMain.Params()
        x = np.log(np.array([80.0, 90.0, 110.0]))
        y = np.log(np.array([85.0, 70.0, 120.0]))
        survival = BarrierMain.single_bridge_survival(
            x, y, math.log(params.lower_barrier), 0.04, lower=True
        )
        self.assertTrue(np.all((0.0 <= survival) & (survival <= 1.0)))
        self.assertEqual(survival[1], 0.0)

        double_survival = BarrierMain.double_bridge_survival(
            x,
            np.log(np.array([90.0, 100.0, 120.0])),
            math.log(params.lower_barrier),
            math.log(params.upper_barrier),
            0.04,
        )
        self.assertTrue(np.all((0.0 <= double_survival) & (double_survival <= 1.0)))

        vanilla = np.array([0.0, 5.0, 20.0])
        knock_out = np.array([0.0, 3.0, 7.0])
        self.assertAllClose(
            BarrierMain.knock_in_from_parity(vanilla, knock_out),
            np.array([0.0, 2.0, 13.0]),
        )

    def test_cliquet_locked_tails_and_bounds(self):
        params = CliquetMain.Params()
        day = 9
        remaining = CliquetMain.remaining_periods(day, params)
        high_accrued = params.global_cap - remaining * params.local_floor + 0.01
        low_accrued = params.global_floor - remaining * params.local_cap - 0.01
        tails = CliquetMain.exact_tail_value(np.array([high_accrued, low_accrued]), day, params)
        low_bound, high_bound = CliquetMain.value_bounds(day, params)
        self.assertAlmostEqual(tails[0], high_bound, places=12)
        self.assertAlmostEqual(tails[1], low_bound, places=12)

        payoff = CliquetMain.payoff_from_accrued(np.array([-1.0, 0.05, 1.0]), params)
        self.assertTrue(np.all(payoff >= params.notional * params.global_floor))
        self.assertTrue(np.all(payoff <= params.notional * params.global_cap))

    def test_slv_cliquet_leverage_and_locked_tails(self):
        params = SLVCliquetMain.Params()
        spots = np.array([35.0, 100.0, 250.0])
        lev = SLVCliquetMain.leverage(spots, params)
        self.assertTrue(np.all((0.5 <= lev) & (lev <= 1.5)))

        month = 9
        remaining = SLVCliquetMain.remaining(month, params)
        high = params.global_cap - remaining * params.local_floor + 0.01
        low = params.global_floor - remaining * params.local_cap - 0.01
        tails = SLVCliquetMain.exact_tail(np.array([high, low]), month, params)
        self.assertAlmostEqual(
            tails[0],
            SLVCliquetMain.discount(month, params) * params.notional * params.global_cap,
            places=12,
        )
        self.assertAlmostEqual(
            tails[1],
            SLVCliquetMain.discount(month, params) * params.notional * params.global_floor,
            places=12,
        )

    def test_basket_asian_correlation_pca_and_terminal_payoff(self):
        params = BasketAsianMain.Params()
        corr = BasketAsianMain.correlation_matrix(params)
        self.assertAllClose(corr, corr.T, atol=1e-12, rtol=1e-12)
        self.assertTrue(np.allclose(np.diag(corr), 1.0))
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(corr))), -1e-10)

        eigenvalues, eigenvectors = BasketAsianMain.pca_loadings(params)
        self.assertTrue(np.all(eigenvalues[:-1] >= eigenvalues[1:] - 1e-12))
        self.assertAllClose(eigenvectors.T @ eigenvectors, np.eye(params.n_assets), atol=1e-10)

        spots = np.full((2, params.n_assets), 100.0)
        running = np.array([11.0 * 95.0, 11.0 * 105.0])
        terminal = BasketAsianMain.moment_lognormal_value(
            spots, running, params.n_fixings - 1, params
        )
        expected = BasketAsianMain.payoff_from_average(
            (running + BasketAsianMain.basket_level(spots)) / params.n_fixings,
            params,
        )
        self.assertAllClose(terminal, expected)

    def test_basket_cliquet_coupon_bounds_and_pca(self):
        params = BasketCliquetMain.Params()
        basis = BasketCliquetMain.pca_basis(params)
        self.assertAllClose(basis.T @ basis, np.eye(3), atol=1e-10)

        returns = np.array(
            [
                [0.10, -0.04, 0.01],
                [-0.08, -0.02, 0.05],
                [0.00, 0.02, 0.03],
            ]
        )
        coupons = BasketCliquetMain.coupon_values(returns, params)
        for variant, values in coupons.items():
            low, high = BasketCliquetMain.coupon_bounds(variant, params)
            self.assertTrue(np.all(values >= low - 1e-12), variant)
            self.assertTrue(np.all(values <= high + 1e-12), variant)

        month = 9
        remaining = BasketCliquetMain.remaining_periods(month, params)
        low_coupon, high_coupon = BasketCliquetMain.coupon_bounds("worst_of", params)
        high_accrued = params.global_cap - remaining * low_coupon + 0.01
        low_accrued = params.global_floor - remaining * high_coupon - 0.01
        tails = BasketCliquetMain.exact_tail(
            np.array([high_accrued, low_accrued]), month, params, "worst_of"
        )
        self.assertAlmostEqual(
            tails[0],
            BasketCliquetMain.discount(month, params) * params.notional * params.global_cap,
            places=12,
        )
        self.assertAlmostEqual(
            tails[1],
            BasketCliquetMain.discount(month, params) * params.notional * params.global_floor,
            places=12,
        )

    def test_autocallable_maturity_redemption(self):
        params = AutocallableMain.Params()
        spots = np.array([50.0, 70.0, 90.0, 130.0])
        values = AutocallableMain.simulate_values(
            spots, params.n_observations, params, 8, params.seed
        )
        expected = np.array(
            [
                params.notional * 50.0 / params.strike,
                params.notional,
                params.notional * (1.0 + params.coupon_per_observation * params.n_observations),
                params.notional * (1.0 + params.coupon_per_observation * params.n_observations),
            ]
        )
        self.assertAllClose(values, expected)

    def test_bermudan_tree_maturity_and_exercise_lower_bound(self):
        original = BermudanMain.BINOMIAL_STEPS_PER_PERIOD
        try:
            BermudanMain.BINOMIAL_STEPS_PER_PERIOD = 12
            params = BermudanMain.Params()
            spots = np.array([70.0, 100.0, 140.0])
            maturity = BermudanMain.benchmark_values(
                spots, params.n_exercise_dates, params
            )
            self.assertAllClose(maturity, BermudanMain.intrinsic(spots, params))
            early = BermudanMain.benchmark_values(spots, 6, params)
            self.assertTrue(np.all(early + 1e-10 >= BermudanMain.intrinsic(spots, params)))
        finally:
            BermudanMain.BINOMIAL_STEPS_PER_PERIOD = original

    def test_random_option_terminal_mc_and_interpolation_knots(self):
        case = RandomOptionMain.PAYOFF_CASES[0]
        market = RandomOptionMain.Market()
        spots = np.array([50.0, 100.0, 175.0])
        self.assertAllClose(
            RandomOptionMain.mc_value(spots, 0.0, market, case, 16, market.seed),
            RandomOptionMain.payoff(spots, case),
        )
        self.assertAllClose(
            RandomOptionMain.payoff(case.strikes, case),
            case.values,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
