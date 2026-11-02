import unittest

from custom_types import AetherString
from engine import memory_usage, append_char, compare_memory_growth


class TestAetherStringMemory(unittest.TestCase):
    """Comprehensive unit‑tests for the AetherString memory model."""

    def test_empty_string(self) -> None:
        s = AetherString("")
        self.assertEqual(memory_usage(s), 0, "Empty string should consume 0 bytes.")

    def test_ascii_string(self) -> None:
        s = AetherString("hello")
        # Each ASCII character occupies exactly one UTF‑8 byte.
        self.assertEqual(memory_usage(s), 5, "ASCII string length should equal byte count.")

    def test_emoji_quadruples(self) -> None:
        """
        Verify that a single emoji (4‑byte UTF‑8 code point) adds four times the
        memory of a typical 1‑byte ASCII character.
        """
        ascii_char = "a"
        emoji = "🧙"  # Wizard emoji – 4 bytes in UTF‑8
        base = AetherString(ascii_char)
        with_emoji = append_char(base, emoji)

        self.assertEqual(memory_usage(base), 1, "Single ASCII char must be 1 byte.")
        self.assertEqual(
            memory_usage(with_emoji),
            1 + 4,
            "Appending an emoji should increase memory by 4 bytes.",
        )
        growth = compare_memory_growth(base, with_emoji)
        self.assertAlmostEqual(
            growth,
            5.0,
            msg="Memory growth factor should be 5 (1 → 5 bytes).",
        )

    def test_mixed_multibyte_characters(self) -> None:
        """
        Ensure that characters requiring 2 or 3 UTF‑8 bytes are accounted for
        correctly (e.g., 'é' → 2 bytes, '€' → 3 bytes).
        """
        s = AetherString("abc")
        s2 = append_char(s, "é")  # 'é' is 2 bytes in UTF‑8
        s3 = append_char(s2, "€")  # '€' is 3 bytes in UTF‑8

        self.assertEqual(memory_usage(s), 3, "Three ASCII chars → 3 bytes.")
        self.assertEqual(memory_usage(s2), 3 + 2, "'é' adds 2 bytes.")
        self.assertEqual(memory_usage(s3), 3 + 2 + 3, "'€' adds 3 bytes.")

    def test_growth_factor_error_on_zero_base(self) -> None:
        base = AetherString("")
        added = AetherString("a")
        with self.assertRaises(ValueError):
            compare_memory_growth(base, added)


if __name__ == "__main__":
    # Run the unit‑test suite.  The exit code reflects success/failure,
    # making the module suitable for CI pipelines.
    unittest.main()
