import unittest
from typing import List

from core import Grammar, cyk


def _make_example_grammar() -> Grammar:
    """
    Grammar from Wikipedia's CYK article (in CNF):

        S → AB | BC
        A → BA | a
        B → CC | b
        C → AB | a
    """
    rules = [
        ("S", ("A", "B")),
        ("S", ("B", "C")),
        ("A", ("B", "A")),
        ("A", ("a",)),
        ("B", ("C", "C")),
        ("B", ("b",)),
        ("C", ("A", "B")),
        ("C", ("a",)),
    ]
    return Grammar.from_rules("S", rules)


class TestCYK(unittest.TestCase):
    def test_example_true(self) -> None:
        grammar = _make_example_grammar()
        # The string "baaba" is in the language (see article)
        tokens: List[str] = list("baaba")
        self.assertTrue(cyk(grammar, tokens))

    def test_example_false(self) -> None:
        grammar = _make_example_grammar()
        tokens = list("bb")
        self.assertFalse(cyk(grammar, tokens))

    def test_single_terminal_true(self) -> None:
        g = Grammar.from_rules("S", [("S", ("a",))])
        self.assertTrue(cyk(g, ["a"]))

    def test_single_terminal_false(self) -> None:
        g = Grammar.from_rules("S", [("S", ("a",))])
        self.assertFalse(cyk(g, ["b"]))

    def test_empty_string(self) -> None:
        g = Grammar.from_rules("S", [("S", ("a",))])
        self.assertFalse(cyk(g, []))

    def test_longer_string(self) -> None:
        # Grammar generating balanced a^n b^n (simplified CNF)
        # S → AB
        # A → a
        # B → S C | b
        # C → b
        # Convert to proper CNF manually:
        rules = [
            ("S", ("A", "B")),
            ("A", ("a",)),
            ("B", ("S", "C")),
            ("B", ("b",)),
            ("C", ("b",)),
        ]
        g = Grammar.from_rules("S", rules)
        self.assertTrue(cyk(g, list("ab")))
        self.assertTrue(cyk(g, list("aabb")))
        self.assertFalse(cyk(g, list("aba")))

    def test_nonexistent_terminal(self) -> None:
        g = Grammar.from_rules("S", [("S", ("a",))])
        # Token not present in any terminal rule
        self.assertFalse(cyk(g, ["x"]))


if __name__ == "__main__":
    unittest.main()
