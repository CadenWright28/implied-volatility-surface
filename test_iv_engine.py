from __future__ import annotations

import math
import unittest

import numpy as np

import iv_engine as iv


class ImpliedVolatilityEngineTests(unittest.TestCase):
    def test_option_side_normalization(self) -> None:
        self.assertEqual(iv.normalize_option_side("c"), "calls")
        self.assertEqual(iv.normalize_option_side("PUT"), "puts")
        with self.assertRaises(ValueError):
            iv.normalize_option_side("straddle")

    def test_black_scholes_call_put_parity(self) -> None:
        spot = 100.0
        strike = 105.0
        rate = 0.04
        t_years = 0.75
        sigma = 0.30

        call = iv._bs_option_snapshot(spot, strike, t_years, sigma, "calls", rate)["price"]
        put = iv._bs_option_snapshot(spot, strike, t_years, sigma, "puts", rate)["price"]
        parity = spot - strike * math.exp(-rate * t_years)
        self.assertAlmostEqual(call - put, parity, places=10)

    def test_implied_volatility_round_trip(self) -> None:
        spot = 52.0
        strike = 55.0
        rate = 0.035
        t_years = 45.0 / 365.0
        sigma = 0.72
        price = iv._bs_option_snapshot(spot, strike, t_years, sigma, "calls", rate)["price"]
        solved = iv._solve_implied_vol_from_price(
            price, spot, strike, t_years, "calls", rate
        )
        self.assertAlmostEqual(solved, sigma, places=6)

    def test_surface_preset_resolution(self) -> None:
        name, band = iv.resolve_surface_moneyness_preset("standard")
        self.assertEqual(name, "standard")
        self.assertEqual(band, (0.50, 2.00))

        name, band = iv.resolve_surface_moneyness_preset("off")
        self.assertEqual(name, "off")
        self.assertIsNone(band)

    def test_mixture_prices_decrease_with_strike(self) -> None:
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        raw_params = np.zeros(3)
        prices = iv.mixture_option_price(
            strikes=strikes,
            t_years=0.5,
            rate=0.04,
            option_side="calls",
            raw_params=raw_params,
            spot=100.0,
            n_components=1,
        )
        self.assertTrue(np.all(np.diff(prices) <= 1e-12))
        self.assertTrue(np.all(prices >= 0.0))


if __name__ == "__main__":
    unittest.main()
