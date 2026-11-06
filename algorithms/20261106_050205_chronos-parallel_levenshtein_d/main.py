"""
main.py
---------

Unit‑test suite and simple benchmark for the stratum‑matrix parallel Levenshtein
implementation defined in core.py.

Running this module executes the tests and prints a brief performance
comparison against the naive implementation.
"""

import random
import string
import time
import unittest
from typing import List

from core import levenshtein_parallel, levenshtein_naive


def _random_string(length: int, alphabet: str = string.ascii_lowercase) -> str:
    """Generate a random string of *length* characters."""
    return ''.join(random.choice(alphabet) for _ in range(length))


class TestLevenshteinParallel(unittest.TestCase):
    """Comprehensive correctness tests."""

    def test_empty_strings(self):
        self.assertEqual(levenshtein_parallel("", ""), 0)
        self.assertEqual(levenshtein_parallel("abc", ""), 3)
        self.assertEqual(levenshtein_parallel("", "xyz"), 3)

    def test_single_character(self):
        self.assertEqual(levenshtein_parallel("a", "a"), 0)
        self.assertEqual(levenshtein_parallel("a", "b"), 1)
        self.assertEqual(levenshtein_parallel("a", ""), 1)
        self.assertEqual(levenshtein_parallel("", "b"), 1)

    def test_known_cases(self):
        cases = [
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("intention", "execution", 5),
            ("algorithm", "altruistic", 6),
        ]
        for s, t, expected in cases:
            with self.subTest(s=s, t=t):
                self.assertEqual(levenshtein_parallel(s, t), expected)

    def test_random_small(self):
        """Compare against naive implementation for many small random strings."""
        for _ in range(200):
            s = _random_string(random.randint(0, 8))
            t = _random_string(random.randint(0, 8))
            expected = levenshtein_naive(s, t)
            result = levenshtein_parallel(s, t)
            self.assertEqual(result, expected,
                             f"Mismatch for s={s!r}, t={t!r}: {result} != {expected}")

    def test_random_medium(self):
        """Medium sized strings (up to 100 chars)."""
        for _ in range(50):
            s = _random_string(random.randint(0, 100))
            t = _random_string(random.randint(0, 100))
            self.assertEqual(levenshtein_parallel(s, t), levenshtein_naive(s, t))

    def test_symmetry(self):
        """Levenshtein distance is symmetric."""
        for _ in range(100):
            s = _random_string(random.randint(0, 30))
            t = _random_string(random.randint(0, 30))
            d1 = levenshtein_parallel(s, t)
            d2 = levenshtein_parallel(t, s)
            self.assertEqual(d1, d2)


def benchmark():
    """Simple benchmark comparing parallel vs naive implementation."""
    sizes = [50, 200, 500, 1000]  # lengths of strings
    repetitions = 5

    print("\nBenchmark (average over {} runs):".format(repetitions))
    print("{:>8} | {:>12} | {:>12}".format("Length", "Parallel(ms)", "Naive(ms)"))
    print("-" * 38)

    for n in sizes:
        s = _random_string(n)
        t = _random_string(n)

        # Warm‑up
        levenshtein_parallel(s, t)
        levenshtein_naive(s, t)

        parallel_times: List[float] = []
        naive_times: List[float] = []

        for _ in range(repetitions):
            start = time.perf_counter()
            levenshtein_parallel(s, t)
            parallel_times.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            levenshtein_naive(s, t)
            naive_times.append((time.perf_counter() - start) * 1000)

        avg_parallel = sum(parallel_times) / repetitions
        avg_naive = sum(naive_times) / repetitions
        print("{:>8} | {:>12.2f} | {:>12.2f}".format(n, avg_parallel, avg_naive))


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a brief benchmark
    benchmark()
