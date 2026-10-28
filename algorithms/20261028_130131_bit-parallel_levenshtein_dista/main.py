"""
Unit tests and demonstration for the bit-parallel Levenshtein distance
implementation.
"""

from __future__ import annotations
import unittest
import random
import string

# Import the core function.
from core import bit_parallel_levenshtein


def levenshtein_naive(a: str, b: str) -> int:
    """
    Naive dynamic programming implementation of Levenshtein distance.
    Used for correctness verification in tests.
    """
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(prev + cost, dp[j - 1] + 1, dp[j] + 1)
            prev = temp
    return dp[m]


class TestLevenshtein(unittest.TestCase):
    """Test suite for the bit-parallel Levenshtein distance implementation."""

    def test_empty_strings(self):
        self.assertEqual(bit_parallel_levenshtein("", ""), 0)
        self.assertEqual(bit_parallel_levenshtein("abc", ""), 3)
        self.assertEqual(bit_parallel_levenshtein("", "xyz"), 3)

    def test_identical_strings(self):
        for s in ["", "a", "abc", "hello world", "😊🚀"]:
            self.assertEqual(bit_parallel_levenshtein(s, s), 0)

    def test_simple_cases(self):
        self.assertEqual(bit_parallel_levenshtein("kitten", "sitting"), 3)
        self.assertEqual(bit_parallel_levenshtein("flaw", "lawn"), 2)
        self.assertEqual(bit_parallel_levenshtein("intention", "execution"), 5)

    def test_unicode(self):
        self.assertEqual(bit_parallel_levenshtein("你好", "你好世界"), 2)
        self.assertEqual(bit_parallel_levenshtein("😊", "😢"), 1)
        self.assertEqual(bit_parallel_levenshtein("a😊b", "a😢b"), 1)

    def test_long_strings(self):
        s = "a" * 1000
        t = "a" * 1000
        self.assertEqual(bit_parallel_levenshtein(s, t), 0)

        s = "a" * 1000
        t = "b" * 1000
        self.assertEqual(bit_parallel_levenshtein(s, t), 1000)

        s = "a" * 999 + "b"
        t = "a" * 1000
        self.assertEqual(bit_parallel_levenshtein(s, t), 1)

    def test_random_strings(self):
        random.seed(0)
        for _ in range(100):
            len_a = random.randint(0, 20)
            len_b = random.randint(0, 20)
            a = "".join(random.choice(string.ascii_lowercase) for _ in range(len_a))
            b = "".join(random.choice(string.ascii_lowercase) for _ in range(len_b))
            expected = levenshtein_naive(a, b)
            result = bit_parallel_levenshtein(a, b)
            self.assertEqual(result, expected,
                             msg=f"Failed for a={a!r}, b={b!r}")

    def test_consistency_with_naive(self):
        """Cross-check against the naive implementation for many cases."""
        random.seed(1)
        for _ in range(200):
            a = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(0, 30)))
            b = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(0, 30)))
            expected = levenshtein_naive(a, b)
            result = bit_parallel_levenshtein(a, b)
            self.assertEqual(result, expected,
                             msg=f"Mismatch: a={a!r}, b={b!r}, expected={expected}, got={result}")


if __name__ == "__main__":
    # Run the unit tests.
    unittest.main(verbosity=2)
    # Demo: compute distance between two sample strings.
    sample_s = "intention"
    sample_t = "execution"
    dist = bit_parallel_levenshtein(sample_s, sample_t)
    print(f"\nLevenshtein distance between '{sample_s}' and '{sample_t}': {dist}")
