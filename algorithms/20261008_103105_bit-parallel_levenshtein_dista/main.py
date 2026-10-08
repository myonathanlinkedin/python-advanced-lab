"""
Unit tests and simple benchmark for the bit‑parallel Levenshtein implementation.
"""

import random
import string
import time
import unittest

from core import levenshtein_bitparallel


def _levenshtein_dp(a: str, b: str) -> int:
    """Classic O(|a|·|b|) dynamic‑programming Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,      # deletion
                current[j - 1] + 1,   # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous = current
    return previous[-1]


class TestLevenshteinBitParallel(unittest.TestCase):
    def test_empty_strings(self):
        self.assertEqual(levenshtein_bitparallel("", ""), 0)
        self.assertEqual(levenshtein_bitparallel("abc", ""), 3)
        self.assertEqual(levenshtein_bitparallel("", "xyz"), 3)

    def test_basic_cases(self):
        cases = [
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("intention", "execution", 5),
            ("abc", "abc", 0),
            ("abc", "acb", 2),
            ("", "a", 1),
            ("a", "", 1),
        ]
        for s, t, d in cases:
            with self.subTest(s=s, t=t):
                self.assertEqual(levenshtein_bitparallel(s, t), d)

    def test_random_small(self):
        for _ in range(200):
            a = ''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 8)))
            b = ''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 8)))
            expected = _levenshtein_dp(a, b)
            result = levenshtein_bitparallel(a, b)
            self.assertEqual(result, expected, f"Failed for {a!r}, {b!r}")

    def test_random_medium(self):
        for _ in range(100):
            a = ''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 30)))
            b = ''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 30)))
            expected = _levenshtein_dp(a, b)
            result = levenshtein_bitparallel(a, b)
            self.assertEqual(result, expected)

    def test_long_strings_consistency(self):
        # Lengths up to 200 characters; DP is still feasible for verification
        for _ in range(20):
            a = ''.join(random.choices(string.ascii_lowercase, k=200))
            b = ''.join(random.choices(string.ascii_lowercase, k=200))
            self.assertEqual(levenshtein_bitparallel(a, b), _levenshtein_dp(a, b))


def benchmark():
    """Simple benchmark comparing bit‑parallel vs DP on medium strings."""
    a = ''.join(random.choices(string.ascii_lowercase, k=500))
    b = ''.join(random.choices(string.ascii_lowercase, k=500))

    start = time.perf_counter()
    for _ in range(100):
        levenshtein_bitparallel(a, b)
    bp_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(10):  # DP is slower; fewer repetitions
        _levenshtein_dp(a, b)
    dp_time = time.perf_counter() - start

    print(f"Bit‑parallel (100 runs): {bp_time:.4f}s")
    print(f"DP (10 runs): {dp_time:.4f}s")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a quick benchmark
    print("\nBenchmark (500‑char strings):")
    benchmark()
