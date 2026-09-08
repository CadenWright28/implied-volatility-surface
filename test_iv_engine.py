import unittest

import numpy as np

import iv_surface as iv


class IVTests(unittest.TestCase):
    def test_side_parser(self):
        self.assertEqual(iv.parse_side("c"), "calls")
        self.assertEqual(iv.parse_side("puts"), "puts")

    def test_call_put_parity(self):
        S, K, T, r, vol = 100, 105, 0.75, 0.04, 0.30
        call = iv.bs_price(S, K, T, r, vol, "calls")
        put = iv.bs_price(S, K, T, r, vol, "puts")
        self.assertAlmostEqual(call - put, S - K*np.exp(-r*T), places=10)

    def test_iv_round_trip(self):
        S, K, T, r, vol = 100, 110, 0.5, 0.04, 0.42
        price = iv.bs_price(S, K, T, r, vol, "calls")
        solved = iv.implied_volatility(price, S, K, T, r, "calls")
        self.assertAlmostEqual(solved, vol, places=8)

    def test_put_iv_round_trip(self):
        S, K, T, r, vol = 100, 90, 0.35, 0.04, 0.55
        price = iv.bs_price(S, K, T, r, vol, "puts")
        solved = iv.implied_volatility(price, S, K, T, r, "puts")
        self.assertAlmostEqual(solved, vol, places=8)

    def test_greeks_are_finite(self):
        greeks = iv.option_greeks(100, 100, 0.5, 0.04, 0.3, "calls")
        self.assertTrue(all(np.isfinite(value) for value in greeks.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
