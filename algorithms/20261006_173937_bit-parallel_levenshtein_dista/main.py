import unittest
import random
import string
import timeit
from typing import List

from core import bit_levenshtein


def dp_levenshtein(s: str, t: str) -> int:
    """Classic dynamic programming implementation of Levenshtein distance."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


class TestBitParallelLevenshtein(unittest.TestCase):
    def test_empty_strings(self):
        self.assertEqual(bit_levenshtein("", ""), 0)
        self.assertEqual(bit_levenshtein("abc", ""), 3)
        self.assertEqual(bit_levenshtein("", "xyz"), 3)

    def test_small_strings(self):
        cases = [
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("intention", "execution", 5),
            ("", "a", 1),
            ("a", "a", 0),
        ]
        for s, t, expected in cases:
            with self.subTest(s=s, t=t):
                self.assertEqual(bit_levenshtein(s, t), expected)

    def test_random_strings(self):
        for _ in range(100):
            n = random.randint(0, 20)
            m = random.randint(0, 20)
            s = "".join(random.choice(string.ascii_lowercase) for _ in range(n))
            t = "".join(random.choice(string.ascii_lowercase) for _ in range(m))
            expected = dp_levenshtein(s, t)
            result = bit_levenshtein(s, t)
            self.assertEqual(result, expected, f"Failed for s={s!r}, t={t!r}")

    def test_max_distance(self):
        s = "abcdefghij"
        t = "abcdefghij"
        self.assertEqual(bit_levenshtein(s, t, max_distance=0), 0)
        self.assertEqual(bit_levenshtein(s, t, max_distance=1), 0)
        self.assertEqual(bit_levenshtein(s, t, max_distance=5), 0)
        # Force early exit
        self.assertEqual(bit_levenshtein("a" * 10, "b" * 10, max_distance=3), 10)

    def test_pattern_length_limit(self):
        long_pattern = "a" * 65
        with self.assertRaises(ValueError):
            bit_levenshtein("abc", long_pattern)


class Benchmark(unittest.TestCase):
    def test_performance(self):
        s = "abcdefghij" * 5
        t = "abcdefghij" * 5
        # Ensure both functions produce same result
        self.assertEqual(bit_levenshtein(s, t), dp_levenshtein(s, t))
        # Measure time
        bit_time = timeit.timeit(
            stmt="bit_levenshtein(s, t)",
            setup="from core import bit_levenshtein; s='abcdefghij'*5; t='abcdefghij'*5",
            number=1000,
        )
        dp_time = timeit.timeit(
            stmt="dp_levenshtein(s, t)",
            setup="from main import dp_levenshtein; s='abcdefghij'*5; t='abcdefghij'*5",
            number=1000,
        )
        print(f"\nBit‑parallel time: {bit_time:.4f}s, DP time: {dp_time:.4f}s")
        # Bit‑parallel should be faster for longer strings
        self.assertLess(bit_time, dp_time)


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demo
    sample_s = "intention"
    sample_t = "execution"
    dist = bit_levenshtein(sample_s, sample_t)
    print(f"\nLevenshtein distance between '{sample_s}' and '{sample_t}': {dist}")
