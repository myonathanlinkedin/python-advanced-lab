"""
Unit‑test suite and simple benchmark for the algorithms in core.py.
Running this file executes the tests and prints a brief demo.
"""

import unittest
from typing import List, Tuple, Dict

from core import (
    lz77_factorization,
    lz77_reconstruct,
    lz_end_factorization,
    lz_end_reconstruct,
    repar_factorization,
    repar_expand,
    factor_count,
    grammar_size,
)


class TestLZ77(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(lz77_factorization(""), [])

    def test_single_char(self):
        self.assertEqual(lz77_factorization("a"), [(0, 0, "a")])

    def test_basic(self):
        text = "aabaabaabaab"
        factors = lz77_factorization(text)
        self.assertEqual(lz77_reconstruct(factors), text)
        # Expected number of factors for this classic example is 4
        self.assertEqual(len(factors), 4)

    def test_no_match(self):
        text = "abcdef"
        factors = lz77_factorization(text)
        self.assertEqual(len(factors), len(text))
        self.assertEqual(lz77_reconstruct(factors), text)


class TestLZEnd(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(lz_end_factorization(""), [])

    def test_single_char(self):
        self.assertEqual(lz_end_factorization("z"), [(0, 0, "z")])

    def test_basic(self):
        text = "aabaabaabaab"
        factors = lz_end_factorization(text)
        self.assertEqual(lz_end_reconstruct(factors), text)
        # LZ‑End never uses more phrases than LZ77 on the same text
        self.assertLessEqual(len(factors), len(lz77_factorization(text)))

    def test_repetitive(self):
        text = "a" * 16
        f77 = lz77_factorization(text)
        fend = lz_end_factorization(text)
        self.assertLessEqual(len(fend), len(f77))


class TestRePair(unittest.TestCase):
    def test_empty(self):
        rules, comp = repar_factorization("")
        self.assertEqual(comp, "")
        self.assertEqual(rules, {})
        self.assertEqual(repar_expand(rules, comp), "")

    def test_no_pair(self):
        text = "abcde"
        rules, comp = repar_factorization(text)
        self.assertEqual(comp, text)
        self.assertEqual(rules, {})
        self.assertEqual(repar_expand(rules, comp), text)

    def test_simple(self):
        text = "ababab"
        rules, comp = repar_factorization(text)
        # Expected: one rule X1 -> ab, compressed X1X1X1
        self.assertIn("X1", rules)
        self.assertEqual(rules["X1"], "ab")
        self.assertEqual(comp, "X1X1X1")
        self.assertEqual(repar_expand(rules, comp), text)

    def test_size_reduction(self):
        text = "a" * 32
        rules, comp = repar_factorization(text)
        self.assertLess(grammar_size(rules, comp), len(text))


def demo() -> None:
    """Simple demonstration printed when the module is executed directly."""
    sample = "aabaabaabaab"
    print("Original:", sample)

    f77 = lz77_factorization(sample)
    print("\nLZ77 factorization (distance, length, next_char):")
    for f in f77:
        print(f)

    print("\nReconstruction matches:",
          lz77_reconstruct(f77) == sample)

    fend = lz_end_factorization(sample)
    print("\nLZ-End factorization:")
    for f in fend:
        print(f)

    print("\nReconstruction matches:",
          lz_end_reconstruct(fend) == sample)

    rules, comp = repar_factorization(sample)
    print("\nRe-Pair grammar rules:")
    for nt, rhs in rules.items():
        print(f"{nt} -> {rhs}")
    print("Compressed string:", comp)
    print("Expansion matches:", repar_expand(rules, comp) == sample)


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n--- Demo ---")
    demo()
