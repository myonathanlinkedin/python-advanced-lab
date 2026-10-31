import unittest
from typing import List, Tuple, Optional

from core import (
    HeaderField,
    DynamicTable,
    HPACKEncoder,
    HPACKDecoder,
    MemorySavingsCalculator,
)


class TestHeaderField(unittest.TestCase):
    def test_size_calculation(self) -> None:
        hf = HeaderField("content-type", "text/html")
        expected = len("content-type") + len("text/html") + 32
        self.assertEqual(hf.size, expected)


class TestDynamicTable(unittest.TestCase):
    def test_add_and_eviction(self) -> None:
        table = DynamicTable(max_size=100)
        h1 = HeaderField("a", "1")
        h2 = HeaderField("b", "2")
        h3 = HeaderField("c", "3")
        table.add(h1)
        self.assertIn(h1, table.entries)
        self.assertEqual(table.current_size, h1.size)

        # Add second entry; may cause eviction depending on size.
        table.add(h2)
        self.assertIn(h2, table.entries)
        self.assertTrue(table.current_size <= table.max_size)

        # Force eviction by adding a large entry.
        large = HeaderField("large-header", "x" * 200)
        table.add(large)
        self.assertIn(large, table.entries)
        self.assertTrue(table.current_size <= table.max_size)
        self.assertNotIn(h1, table.entries)  # oldest should be evicted.

    def test_get_by_index(self) -> None:
        table = DynamicTable()
        h = HeaderField("d", "4")
        table.add(h)
        self.assertIs(table.get(1), h)
        with self.assertRaises(IndexError):
            table.get(0)


class TestHPACKRoundTrip(unittest.TestCase):
    def test_encode_decode(self) -> None:
        headers = [
            HeaderField(":method", "GET"),
            HeaderField("custom", "value"),
            HeaderField(":path", "/index.html"),
            HeaderField("custom", "value"),  # duplicate to test indexing
        ]
        encoder = HPACKEncoder()
        encoded = encoder.encode(headers)

        decoder = HPACKDecoder()
        decoded = decoder.decode(encoded)

        self.assertEqual(decoded, headers)

    def test_indexed_representation(self) -> None:
        # The first header is in the static table, should be indexed.
        headers = [HeaderField(":method", "GET")]
        encoder = HPACKEncoder()
        encoded = encoder.encode(headers)
        self.assertEqual(encoded, [(2, None)])  # static index 2 per _STATIC_TABLE

        # After decoding, the same header should still be indexed.
        decoder = HPACKDecoder()
        decoded = decoder.decode(encoded)
        self.assertEqual(decoded, headers)


class TestMemorySavings(unittest.TestCase):
    def test_savings_basic(self) -> None:
        headers = [
            HeaderField(":method", "GET"),
            HeaderField("custom", "value"),
            HeaderField(":path", "/"),
        ]
        encoder = HPACKEncoder()
        encoded = encoder.encode(headers)

        naive = MemorySavingsCalculator.naive_size(headers)
        encoded_sz = MemorySavingsCalculator.encoded_size(encoded)
        savings = MemorySavingsCalculator.savings(headers, encoded)

        # Verify calculations are consistent.
        self.assertEqual(savings, naive - encoded_sz)
        # Expect at least some savings because the first header is indexed.
        self.assertGreaterEqual(savings, 0)

    def test_no_savings_when_all_literals(self) -> None:
        # Use headers unlikely to be in static table.
        headers = [
            HeaderField("x-foo", "a"),
            HeaderField("x-bar", "b"),
        ]
        encoder = HPACKEncoder()
        encoded = encoder.encode(headers)

        # All entries should be literals (index 0).
        self.assertTrue(all(idx == 0 for idx, _ in encoded))
        savings = MemorySavingsCalculator.savings(headers, encoded)
        # With our simple model, literals incur a +2 overhead, so savings may be negative.
        self.assertLessEqual(savings, 0)


class TestEncoderEvictionPolicy(unittest.TestCase):
    def test_dynamic_table_clears_on_oversize_entry(self) -> None:
        table = DynamicTable(max_size=50)
        encoder = HPACKEncoder(dynamic_table=table)

        # Header that fits.
        small = HeaderField("a", "b")
        encoder.encode([small])
        self.assertIn(small, table.entries)

        # Header too large to ever fit.
        huge = HeaderField("huge", "x" * 200)
        encoder.encode([huge])
        self.assertEqual(len(table), 0)  # table cleared per spec.


if __name__ == "__main__":
    unittest.main()
