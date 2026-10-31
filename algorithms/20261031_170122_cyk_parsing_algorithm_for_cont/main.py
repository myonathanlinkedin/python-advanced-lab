import unittest
import time
from typing import List

from core import Grammar, cyk_parse


def build_simple_ab_grammar() -> Grammar:
    """
    Grammar generating any number (including zero) of the substring "ab".
    CNF:
        S -> AB | ε   (ε handled by allowing S to derive empty via a special rule)
        A -> a
        B -> b
    Since pure CNF forbids ε, we model the empty string by allowing S to derive AB repeatedly
    and treat the empty input as not accepted. For testing we only use non‑empty strings.
    """
    g = Grammar(start="S")
    g.add_production("S", ("A", "B"))   # S -> AB
    g.add_production("S", ("S", "S"))   # S -> SS (allows concatenation of AB blocks)
    g.add_production("A", ("a",))       # A -> a
    g.add_production("B", ("b",))       # B -> b
    return g


class TestCYK(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = build_simple_ab_grammar()

    def assertCYK(self, tokens: List[str], expected: bool) -> None:
        result = cyk_parse(self.grammar, tokens)
        self.assertEqual(result, expected,
                         f"CYK result for {tokens!r} expected {expected} got {result}")

    def test_single_ab(self):
        self.assertCYK(["a", "b"], True)

    def test_double_ab(self):
        self.assertCYK(["a", "b", "a", "b"], True)

    def test_triple_ab(self):
        self.assertCYK(["a", "b", "a", "b", "a", "b"], True)

    def test_invalid_missing_b(self):
        self.assertCYK(["a", "a", "b"], False)

    def test_invalid_extra_a(self):
        self.assertCYK(["a", "b", "a"], False)

    def test_nonmember_long(self):
        self.assertCYK(["a", "b", "b", "a"], False)

    def test_empty_string(self):
        # Grammar does not generate ε in strict CNF, expect False.
        self.assertCYK([], False)

    def test_performance(self):
        # Generate a long valid string (100 "ab" pairs) and ensure it parses quickly.
        tokens = ["a", "b"] * 100
        start = time.perf_counter()
        self.assertCYK(tokens, True)
        duration = time.perf_counter() - start
        self.assertLess(duration, 0.5, "CYK parsing took too long for 200 symbols")


if __name__ == "__main__":
    unittest.main()
