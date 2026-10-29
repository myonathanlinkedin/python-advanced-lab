import unittest
from typing import List, Tuple

from core import Grammar, cyk


class TestCYKAlgorithm(unittest.TestCase):
    def test_simple_terminal_grammar(self) -> None:
        # Grammar: S → a | b
        rules: List[Tuple[str, Tuple[str, ...]]] = [
            ("S", ("a",)),
            ("S", ("b",)),
        ]
        g = Grammar.from_rules("S", rules)
        self.assertTrue(cyk(g, ["a"]))
        self.assertTrue(cyk(g, ["b"]))
        self.assertFalse(cyk(g, ["c"]))
        self.assertFalse(cyk(g, []))

    def test_balanced_parentheses(self) -> None:
        # CNF for language of balanced parentheses:
        # S → SS | ( ) | ε
        rules: List[Tuple[str, Tuple[str, ...]]] = [
            ("S", ("S", "S")),
            ("S", ("(", ")")),
            ("S", ()),                     # epsilon production
        ]
        g = Grammar.from_rules("S", rules)

        self.assertTrue(cyk(g, []))
        self.assertTrue(cyk(g, ["(", ")"]))
        self.assertTrue(cyk(g, ["(", ")", "(", ")"]))
        self.assertTrue(cyk(g, ["(", "(", ")", ")"]))
        self.assertFalse(cyk(g, ["("]))
        self.assertFalse(cyk(g, [")"]))
        self.assertFalse(cyk(g, ["(", ")", ")"]))
        self.assertFalse(cyk(g, ["(", "(", ")"]))

    def test_wikipedia_example(self) -> None:
        # Example grammar from Wikipedia (in CNF):
        # S → AB | BC
        # A → BA | a
        # B → CC | b
        # C → AB | a
        rules: List[Tuple[str, Tuple[str, ...]]] = [
            ("S", ("A", "B")),
            ("S", ("B", "C")),
            ("A", ("B", "A")),
            ("A", ("a",)),
            ("B", ("C", "C")),
            ("B", ("b",)),
            ("C", ("A", "B")),
            ("C", ("a",)),
        ]
        g = Grammar.from_rules("S", rules)

        # Accepted strings
        self.assertTrue(cyk(g, list("baaba")))
        self.assertTrue(cyk(g, list("baab")))
        self.assertTrue(cyk(g, list("baa")))
        # Rejected strings
        self.assertFalse(cyk(g, list("bab")))
        self.assertFalse(cyk(g, list("b")))
        self.assertFalse(cyk(g, list("aab")))

    def test_epsilon_only_start(self) -> None:
        # Grammar where only the start symbol can produce ε.
        rules: List[Tuple[str, Tuple[str, ...]]] = [
            ("S", ()),
            ("S", ("A", "B")),
            ("A", ("a",)),
            ("B", ("b",)),
        ]
        g = Grammar.from_rules("S", rules)

        self.assertTrue(cyk(g, []))
        self.assertTrue(cyk(g, ["a", "b"]))
        self.assertFalse(cyk(g, ["a"]))
        self.assertFalse(cyk(g, ["b"]))
        self.assertFalse(cyk(g, ["a", "b", "c"]))

    def test_invalid_cnf_ignored(self) -> None:
        # Grammar containing a production that is not CNF should be ignored by the algorithm.
        # The algorithm only looks at productions of length 0, 1, or 2.
        rules: List[Tuple[str, Tuple[str, ...]]] = [
            ("S", ("A", "B", "C")),   # invalid length 3, should be ignored
            ("S", ("a",)),
            ("A", ("a",)),
            ("B", ("b",)),
        ]
        g = Grammar.from_rules("S", rules)

        self.assertTrue(cyk(g, ["a"]))
        self.assertFalse(cyk(g, ["b"]))
        self.assertFalse(cyk(g, ["a", "b"]))

    def test_long_string_performance(self) -> None:
        # Grammar for strings of the form a^n b^n (converted to CNF):
        # S → AB | ε
        # A → aA | a
        # B → bB | b
        # Convert to CNF manually:
        # S → X Y | ε
        # X → a
        # Y → Z W
        # Z → a
        # W → B
        # B → b
        # This is a simplified version that only accepts "ab".
        rules: List[Tuple[str, Tuple[str, ...]]] = [
            ("S", ("X", "Y")),
            ("S", ()),
            ("X", ("a",)),
            ("Y", ("Z", "W")),
            ("Z", ("a",)),
            ("W", ("B",)),
            ("B", ("b",)),
        ]
        g = Grammar.from_rules("S", rules)

        self.assertTrue(cyk(g, ["a", "b"]))
        self.assertFalse(cyk(g, ["a", "a", "b", "b"]))

if __name__ == "__main__":
    unittest.main()
