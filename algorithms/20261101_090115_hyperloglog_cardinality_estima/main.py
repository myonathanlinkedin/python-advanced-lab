import unittest
import random
from typing import Set

from core import HyperLogLog


class TestHyperLogLog(unittest.TestCase):
    def test_empty(self):
        hll = HyperLogLog(p=10)
        self.assertEqual(len(hll), 0)

    def test_single_element(self):
        hll = HyperLogLog(p=10)
        hll.add("apple")
        self.assertEqual(len(hll), 1)

    def test_duplicate_elements(self):
        hll = HyperLogLog(p=10)
        for _ in range(10):
            hll.add("banana")
        self.assertEqual(len(hll), 1)

    def test_small_cardinality_accuracy(self):
        """For n ≤ 1000 the relative error should be < 10%."""
        for n in [10, 50, 100, 500, 1000]:
            with self.subTest(n=n):
                hll = HyperLogLog(p=12)
                for i in range(n):
                    hll.add(i)
                estimate = len(hll)
                rel_error = abs(estimate - n) / n
                self.assertLess(rel_error, 0.10)

    def test_large_cardinality_accuracy(self):
        """For n ≈ 100 000 the relative error should be < 5%."""
        n = 100_000
        hll = HyperLogLog(p=14)
        for i in range(n):
            hll.add(i)
        estimate = len(hll)
        rel_error = abs(estimate - n) / n
        self.assertLess(rel_error, 0.05)

    def test_merge(self):
        """Merging two sketches should give the same estimate as a single sketch."""
        n = 50_000
        items = list(range(n))
        random.shuffle(items)

        h1 = HyperLogLog(p=13)
        h2 = HyperLogLog(p=13)

        # Split items roughly in half
        for x in items[: n // 2]:
            h1.add(x)
        for x in items[n // 2 :]:
            h2.add(x)

        merged = HyperLogLog(p=13)
        merged.merge(h1)
        merged.merge(h2)

        # Reference sketch built from all items
        reference = HyperLogLog(p=13)
        for x in items:
            reference.add(x)

        self.assertAlmostEqual(len(merged), len(reference), delta=0.05 * n)

    def test_merge_incompatible_precision(self):
        h1 = HyperLogLog(p=10)
        h2 = HyperLogLog(p=12)
        with self.assertRaises(ValueError):
            h1.merge(h2)

    def test_type_errors(self):
        h = HyperLogLog(p=10)
        with self.assertRaises(TypeError):
            h.merge(42)  # not a HyperLogLog instance


def benchmark():
    """Simple benchmark printing operations per second."""
    import time

    n = 200_000
    hll = HyperLogLog(p=14)
    start = time.perf_counter()
    for i in range(n):
        hll.add(i)
    duration = time.perf_counter() - start
    ops = n / duration
    print(f"Inserted {n} items in {duration:.3f}s ({ops:,.0f} ops/s).")
    print(f"Estimated cardinality: {len(hll)}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a quick benchmark
    print("\n--- Benchmark ---")
    benchmark()
