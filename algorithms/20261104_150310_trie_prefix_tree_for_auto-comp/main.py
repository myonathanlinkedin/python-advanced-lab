import unittest
from typing import List, Tuple

# Import the Trie implementation from core.py
from core import Trie


class TestTrie(unittest.TestCase):
    def setUp(self) -> None:
        self.trie = Trie()
        words = [
            ("apple", 5),
            ("app", 3),
            ("application", 2),
            ("apt", 4),
            ("banana", 7),
            ("band", 1),
            ("bandana", 2),
            ("cat", 6),
            ("cater", 1),
            ("cattle", 3),
        ]
        for w, f in words:
            self.trie.insert(w, f)

    def test_insert_and_frequency(self) -> None:
        self.assertEqual(self.trie.get_frequency("apple"), 5)
        self.assertEqual(self.trie.get_frequency("app"), 3)
        self.assertEqual(self.trie.get_frequency("nonexistent"), 0)

        # Increment existing word
        self.trie.insert("apple", 2)
        self.assertEqual(self.trie.get_frequency("apple"), 7)

    def test_autocomplete_basic(self) -> None:
        result = self.trie.autocomplete("app", k=3)
        expected = [("apple", 5), ("app", 3), ("application", 2)]
        self.assertEqual(result, expected)

    def test_autocomplete_tie_break_lexicographic(self) -> None:
        # Insert words with same frequency to test lexicographic ordering
        self.trie.insert("apex", 5)
        self.trie.insert("apricot", 5)
        result = self.trie.autocomplete("ap", k=5)
        # Expected order: frequency desc (5), then alphabetical
        expected_words = ["apex", "apple", "apricot", "app", "application"]
        self.assertEqual([w for w, _ in result], expected_words[: len(result)])

    def test_autocomplete_empty_prefix(self) -> None:
        # Should return the most frequent words overall
        result = self.trie.autocomplete("", k=4)
        expected = [("banana", 7), ("apple", 5), ("apt", 4), ("cat", 6)]
        # Note: "cat" has freq 6, should appear second
        expected = [("banana", 7), ("cat", 6), ("apple", 5), ("apt", 4)]
        self.assertEqual(result, expected)

    def test_autocomplete_nonexistent_prefix(self) -> None:
        self.assertEqual(self.trie.autocomplete("zzz"), [])

    def test_autocomplete_k_larger_than_available(self) -> None:
        result = self.trie.autocomplete("ban", k=10)
        expected = [("banana", 7), ("bandana", 2), ("band", 1)]
        self.assertEqual(result, expected)

    def test_unicode_support(self) -> None:
        self.trie.insert("naïve", 3)
        self.trie.insert("naïveté", 2)
        result = self.trie.autocomplete("naï", k=2)
        expected = [("naïve", 3), ("naïveté", 2)]
        self.assertEqual(result, expected)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(TypeError):
            self.trie.insert(123)  # type: ignore
        with self.assertRaises(ValueError):
            self.trie.insert("test", 0)
        with self.assertRaises(TypeError):
            self.trie.autocomplete(456)  # type: ignore
        with self.assertRaises(ValueError):
            self.trie.autocomplete("test", k=0)


def benchmark_autocomplete(trie: Trie, prefix: str, k: int, repeats: int = 1000) -> float:
    """
    Simple micro‑benchmark: time taken to perform *repeats* autocomplete queries.
    Returns average time in microseconds.
    """
    import time

    start = time.perf_counter()
    for _ in range(repeats):
        trie.autocomplete(prefix, k)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1_000_000  # µs


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demonstration & tiny benchmark
    demo_trie = Trie()
    sample_words = ["dog", "deer", "deal", "debug", "delta", "demand"]
    for w in sample_words:
        demo_trie.insert(w, 1)

    print("\nDemo autocomplete for prefix 'de' (top 3):")
    for word, freq in demo_trie.autocomplete("de", k=3):
        print(f"{word} (freq={freq})")

    avg_us = benchmark_autocomplete(demo_trie, "de", k=3, repeats=5000)
    print(f"\nAverage autocomplete latency: {avg_us:.2f} µs")
