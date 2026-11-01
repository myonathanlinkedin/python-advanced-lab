"""
main.py
Unit‑test suite for the fast signed 32‑bit division implementation.
Run the module directly to execute tests.
"""

import unittest
import random
from core import fast_divide, fast_mod, INT_MIN, INT_MAX


def _c_trunc_div(a: int, b: int) -> int:
    """Emulate C's trunc‑toward‑zero division for signed 32‑bit integers."""
    # Python's int(a / b) truncates toward zero.
    return int(a / b)


def _c_trunc_mod(a: int, b: int) -> int:
    """Emulate C's remainder (a - (a/b)*b) with trunc division."""
    return a - _c_trunc_div(a, b) * b


class TestFastDivision(unittest.TestCase):
    def test_edge_divisors(self):
        # Test special divisors: 1, -1, powers of two, INT_MIN, INT_MAX
        divisors = [1, -1, 2, -2, 4, -4, 8, -8, INT_MIN, INT_MAX]
        for d in divisors:
            for n in [INT_MIN, -123456789, -1, 0, 1, 123456789, INT_MAX]:
                with self.subTest(divisor=d, dividend=n):
                    q = fast_divide(n, d)
                    r = fast_mod(n, d)
                    self.assertEqual(q, _c_trunc_div(n, d))
                    self.assertEqual(r, _c_trunc_mod(n, d))

    def test_random_pairs(self):
        random.seed(0)
        for _ in range(5000):
            d = random.randint(INT_MIN, INT_MAX)
            # Skip zero divisor
            if d == 0:
                continue
            n = random.randint(INT_MIN, INT_MAX)
            q = fast_divide(n, d)
            r = fast_mod(n, d)
            self.assertEqual(q, _c_trunc_div(n, d))
            self.assertEqual(r, _c_trunc_mod(n, d))

    def test_consistency(self):
        # Verify that fast_mod matches n - q*d for a range of values
        for d in range(-100, 101):
            if d == 0:
                continue
            for n in range(-200, 201):
                q = fast_divide(n, d)
                r = fast_mod(n, d)
                self.assertEqual(r, n - q * d)

    def test_overflow_behavior(self):
        # Ensure that INT_MIN / -1 does not raise in Python (unlike C)
        self.assertEqual(fast_divide(INT_MIN, -1), -INT_MIN)

    def test_cache_reuse(self):
        # Ensure that repeated calls reuse the cached MagicDivider
        from core import _divider_cache
        _divider_cache.clear()
        d = 7
        fast_divide(10, d)
        self.assertIn(d, _divider_cache)
        old_obj = _divider_cache[d]
        fast_divide(20, d)
        self.assertIs(old_obj, _divider_cache[d])


if __name__ == "__main__":
    unittest.main(verbosity=2)
