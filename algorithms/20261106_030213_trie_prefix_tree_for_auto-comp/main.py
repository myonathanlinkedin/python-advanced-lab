import unittest
from typing import List, Tuple

from core import Trie


class TestTrie(unittest.TestCase):
    def setUp(self) -> None:
        self.trie = Trie()
        # Populate with a deterministic dataset
        words = [
            ("apple", 5),
            ("app", 3),
            ("application", 2),
            ("apt", 4),
            ("banana", 7),
            ("band", 1),
            ("bandana", 2),
            ("bandit", 3),
            ("cat", 6),
            ("cater", 1),
            ("cattle", 2),
        ]
        for w, f in words:
            self.trie.insert(w, f)

    def test_insert_and_contains(self) -> None:
        self.assertIn("apple", self.trie)
        self.assertIn("bandana", self.trie)
        self.assertNotIn("unknown", self.trie)

    def test_frequency_accumulation(self) -> None:
        # Insert same word again and verify cumulative frequency
        self.trie.insert("apple", 2)
        self.assertEqual(self.trie.get_frequency("apple"), 7)

        # Increment via dedicated method
        self.trie.increment("banana", 3)
        self.assertEqual(self.trie.get_frequency("banana"), 10)

    def test_len(self) -> None:
        # Initial distinct words count
        self.assertEqual(len(self.trie), 11)
        # Adding a new word increases count
        self.trie.insert("dog", 1)
        self.assertEqual(len(self.trie), 12)
        # Inserting an existing word does not change count
        self.trie.insert("cat", 5)
        self.assertEqual(len(self.trie), 12)

    def test_autocomplete_basic(self) -> None:
        # Top 3 completions for prefix "ap"
        result = self.trie.autocomplete("ap", k=3)
        expected = [("apple", 5), ("apt", 4), ("app", 3)]
        self.assertEqual(result, expected)

    def test_autocomplete_tie_break(self) -> None:
        # Create tie in frequency
        self.trie.insert("apex", 4)   # same freq as "apt"
        result = self.trie.autocomplete("ap", k=4)
        # Expected order: apple(5), apex(4), apt(4), app(3) – lexical tie between apex & apt
        expected = [("apple", 5), ("apex", 4), ("apt", 4), ("app", 3)]
        self.assertEqual(result, expected)

    def test_autocomplete_empty_prefix(self) -> None:
        # Global top‑5 words
        result = self.trie.autocomplete("", k=5)
        expected = [("banana", 7), ("cat", 6), ("apple", 5), ("apt", 4), ("app", 3)]
        self.assertEqual(result, expected)

    def test_autocomplete_nonexistent_prefix(self) -> None:
        self.assertEqual(self.trie.autocomplete("zz", k=5), [])

    def test_autocomplete_k_exceeds(self) -> None:
        # Request more suggestions than exist for a prefix
        result = self.trie.autocomplete("ban", k=10)
        expected = [("banana", 7), ("bandit", 3), ("bandana", 2), ("band", 1)]
        self.assertEqual(result, expected)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            self.trie.insert("", 1)
        with self.assertRaises(ValueError):
            self.trie.insert("test", 0)
        with self.assertRaises(ValueError):
            self.trie.autocomplete("a", k=-1)


if __name__ == "__main__":
    # Run unit tests with verbose output
    unittest.main(verbosity=2)
