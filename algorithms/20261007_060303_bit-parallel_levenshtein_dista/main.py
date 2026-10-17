import unittest
import random
import string
import time
from typing import Tuple

from core import bit_parallel_levenshtein


def reference_levenshtein(a: str, b: str) -> int:
    """Standard dynamic programming implementation for validation."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


class TestBitParallelLevenshtein(unittest.TestCase):
    def test_empty_strings(self):
        self.assertEqual(bit_parallel_levenshtein("", ""), 0)
        self.assertEqual(bit_parallel_levenshtein("abc", ""), 3)
        self.assertEqual(bit_parallel_levenshtein("", "xyz"), 3)

    def test_identical_strings(self):
        for s in ["", "a", "abc", "hello world"]:
            self.assertEqual(bit_parallel_levenshtein(s, s), 0)

    def test_random_strings(self):
        for _ in range(100):
            a = "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(0, 20)))
            b = "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(0, 20)))
            expected = reference_levenshtein(a, b)
            result = bit_parallel_levenshtein(a, b)
            self.assertEqual(result, expected, f"Failed for a={a!r}, b={b!r}")

    def test_pattern_length_limit(self):
        long_pattern = "a" * 65
        with self.assertRaises(ValueError):
            bit_parallel_levenshtein("abc", long_pattern)

    def test_consistency_with_reference(self):
        for _ in range(50):
            a = "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(0, 64)))
            b = "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(0, 64)))
            expected = reference_levenshtein(a, b)
            result = bit_parallel_levenshtein(a, b)
            self.assertEqual(result, expected)


def benchmark() -> Tuple[float, float]:
    """Simple benchmark comparing bit‑parallel and reference implementations."""
    a = "the quick brown fox jumps over the lazy dog"
    b = "the quick brown fox jumps over the lazy cat"
    iterations = 10000

    start = time.perf_counter()
    for _ in range(iterations):
        bit_parallel_levenshtein(a, b)
    bit_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        reference_levenshtein(a, b)
    ref_time = time.perf_counter() - start

    return bit_time, ref_time


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demo and benchmark
    demo_a = "kitten"
    demo_b = "sitting"
    print(f"\nDemo: Levenshtein('{demo_a}', '{demo_b}') = {bit_parallel_levenshtein(demo_a, demo_b)}")
    bit_t, ref_t = benchmark()
    print(f"\nBenchmark over 10,000 iterations:")
    print(f"  Bit‑parallel time: {bit_t:.4f}s")
    print(f"  Reference time:    {ref_t:.4f}s")
    print(f"  Speedup: {ref_t / bit_t:.2f}x")
