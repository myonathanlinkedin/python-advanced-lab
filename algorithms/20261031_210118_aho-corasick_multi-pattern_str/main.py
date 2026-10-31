import unittest
import random
import string
import time
from typing import List, Tuple

from core import AhoCorasickAutomaton


def brute_force_search(patterns: List[str], text: str) -> List[Tuple[int, str]]:
    """Reference implementation using naive scanning."""
    matches: List[Tuple[int, str]] = []
    for pat in patterns:
        if not pat:
            continue
        start = 0
        while True:
            idx = text.find(pat, start)
            if idx == -1:
                break
            matches.append((idx + len(pat) - 1, pat))
            start = idx + 1
    matches.sort()
    return matches


class TestAhoCorasick(unittest.TestCase):
    def setUp(self) -> None:
        self.ac = AhoCorasickAutomaton()

    def test_single_pattern(self) -> None:
        pat = "abc"
        self.ac.add_pattern(pat)
        self.ac.build()
        text = "xxabcxxababc"
        expected = [(4, pat), (10, pat)]
        result = list(self.ac.search(text))
        self.assertEqual(result, expected)

    def test_multiple_patterns(self) -> None:
        patterns = ["he", "she", "his", "hers"]
        for p in patterns:
            self.ac.add_pattern(p)
        self.ac.build()
        text = "ushers"
        expected = brute_force_search(patterns, text)
        result = list(self.ac.search(text))
        self.assertEqual(result, expected)

    def test_overlapping_patterns(self) -> None:
        patterns = ["a", "aa", "aaa"]
        for p in patterns:
            self.ac.add_pattern(p)
        self.ac.build()
        text = "aaaa"
        expected = brute_force_search(patterns, text)
        result = list(self.ac.search(text))
        self.assertEqual(result, expected)

    def test_no_match(self) -> None:
        patterns = ["xyz", "123"]
        for p in patterns:
            self.ac.add_pattern(p)
        self.ac.build()
        text = "abcdef"
        self.assertEqual(list(self.ac.search(text)), [])

    def test_empty_text(self) -> None:
        patterns = ["a", "b"]
        for p in patterns:
            self.ac.add_pattern(p)
        self.ac.build()
        self.assertEqual(list(self.ac.search("")), [])

    def test_empty_patterns(self) -> None:
        # Adding empty strings should be ignored
        self.ac.add_pattern("")
        self.ac.add_pattern("")
        self.ac.build()
        self.assertEqual(len(self.ac), 0)
        self.assertEqual(list(self.ac.search("anything")), [])

    def test_unicode_support(self) -> None:
        patterns = ["naïve", "café", "résumé"]
        for p in patterns:
            self.ac.add_pattern(p)
        self.ac.build()
        text = "My résumé includes a naïve café."
        expected = brute_force_search(patterns, text)
        result = list(self.ac.search(text))
        self.assertEqual(result, expected)

    def test_randomized(self) -> None:
        random.seed(0)
        for _ in range(20):
            # Generate random patterns
            patterns = [
                "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 5)))
                for _ in range(random.randint(1, 10))
            ]
            # Ensure uniqueness to simplify verification
            patterns = list(dict.fromkeys(patterns))
            ac = AhoCorasickAutomaton()
            for p in patterns:
                ac.add_pattern(p)
            ac.build()
            # Random text
            text = "".join(random.choices(string.ascii_lowercase, k=200))
            expected = brute_force_search(patterns, text)
            result = list(ac.search(text))
            self.assertEqual(result, expected)

    def test_performance_small(self) -> None:
        # Simple benchmark to ensure linear behaviour (not a strict timing test)
        patterns = ["ab", "bc", "abc", "bca"]
        for p in patterns:
            self.ac.add_pattern(p)
        self.ac.build()
        text = "abcabcbabcabca" * 1000
        start = time.perf_counter()
        list(self.ac.search(text))
        duration = time.perf_counter() - start
        # Expect sub‑second for this modest workload
        self.assertLess(duration, 0.5)


if __name__ == "__main__":
    unittest.main()
