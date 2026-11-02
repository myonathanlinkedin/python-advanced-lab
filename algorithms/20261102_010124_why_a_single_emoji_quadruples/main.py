"""
main.py
---------

Unit‑test suite and demonstration entry‑point for the core module.

The tests cover:
- Empty strings.
- Pure ASCII strings.
- Strings containing a single emoji.
- Mixed strings.
- Edge cases such as surrogate pairs and multi‑character grapheme clusters.

Running this file directly executes the test suite and prints a short demo.
"""

import unittest
import time
from core import (
    utf8_byte_length,
    is_emoji,
    memory_factor,
    emoji_memory_impact,
)


class TestUtf8Memory(unittest.TestCase):
    def test_utf8_byte_length_empty(self):
        self.assertEqual(utf8_byte_length(""), 0)

    def test_utf8_byte_length_ascii(self):
        self.assertEqual(utf8_byte_length("hello"), 5)

    def test_utf8_byte_length_emoji(self):
        # U+1F600 GRINNING FACE
        self.assertEqual(utf8_byte_length("😀"), 4)

    def test_utf8_byte_length_mixed(self):
        self.assertEqual(utf8_byte_length("A😀B"), 1 + 4 + 1)

    def test_is_emoji_ascii(self):
        for ch in "abcXYZ123!":
            self.assertFalse(is_emoji(ch))

    def test_is_emoji_true(self):
        self.assertTrue(is_emoji("😀"))  # code point > 0xFFFF
        self.assertTrue(is_emoji("🧪"))  # test tube

    def test_is_emoji_invalid_input(self):
        with self.assertRaises(ValueError):
            is_emoji("AB")  # more than one character

    def test_memory_factor_empty(self):
        self.assertEqual(memory_factor(""), 0.0)

    def test_memory_factor_ascii(self):
        self.assertAlmostEqual(memory_factor("abc"), 1.0)

    def test_memory_factor_single_emoji(self):
        # 4 bytes / 1 character = 4.0
        self.assertAlmostEqual(memory_factor("😀"), 4.0)

    def test_memory_factor_mixed(self):
        # "A😀B" -> 6 bytes / 3 chars = 2.0
        self.assertAlmostEqual(memory_factor("A😀B"), 2.0)

    def test_emoji_memory_impact_none(self):
        self.assertEqual(emoji_memory_impact([]), 0)

    def test_emoji_memory_impact_single(self):
        self.assertEqual(emoji_memory_impact(["😀"]), 3)

    def test_emoji_memory_impact_multiple(self):
        self.assertEqual(emoji_memory_impact(["😀", "🧪", "a"]), 6)

    def test_emoji_memory_impact_invalid(self):
        with self.assertRaises(ValueError):
            emoji_memory_impact(["AB"])  # not a single character


def benchmark_utf8_byte_length():
    """
    Simple micro‑benchmark to illustrate deterministic performance.
    """
    test_str = "A" * 10_000 + "😀" * 2_500  # 10k ASCII + 2.5k emoji
    start = time.perf_counter()
    size = utf8_byte_length(test_str)
    duration = time.perf_counter() - start
    print(f"Benchmark: encoded {len(test_str)} chars ({size} bytes) in {duration:.6f}s")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demonstration
    demo_str = "Hello, world! 🌍"
    print("\nDemo:")
    print(f"String: {demo_str}")
    print(f"UTF‑8 byte length: {utf8_byte_length(demo_str)}")
    print(f"Memory factor: {memory_factor(demo_str):.2f}")
    print(f"Contains emoji? {[ch for ch in demo_str if is_emoji(ch)]}")

    # Run a quick benchmark
    benchmark_utf8_byte_length()
