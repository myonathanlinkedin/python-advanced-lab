"""
Unit tests and simple demonstration for the HyperLogLog implementation.
"""

import unittest
import random
from core import HyperLogLog


class TestHyperLogLog(unittest.TestCase):
    def test_empty(self) -> None:
        hll = HyperLogLog(p=10)  # 1024 registers
        self.assertEqual(hll.count(), 0)

    def test_single_element(self) -> None:
        hll = HyperLogLog(p=10)
        hll.add("unique")
        self.assertEqual(hll.count(), 1)

    def test_small_cardinality(self) -> None:
        hll = HyperLogLog(p=10)
        elements = list(range(10))
        hll.update(elements)
        # Linear counting should be exact for such a small set
        self.assertEqual(hll.count(), 10)

    def test_accuracy_medium(self) -> None:
        # Expect ~1% error for p=14 (16384 registers)
        p = 14
        hll = HyperLogLog(p=p)
        true_n = 100_000
        hll.update(range(true_n))
        estimate = hll.count()
        error = abs(estimate - true_n) / true_n
        self.assertLessEqual(error, 0.02)  # 2 % tolerance

    def test_merge(self) -> None:
        p = 12
        hll1 = HyperLogLog(p=p)
        hll2 = HyperLogLog(p=p)
        set1 = {f"a{i}" for i in range(5000)}
        set2 = {f"b{i}" for i in range(5000)}
        hll1.update(set1)
        hll2.update(set2)
        hll1.merge(hll2)
        estimate = hll1.count()
        true_n = len(set1 | set2)
        error = abs(estimate - true_n) / true_n
        self.assertLessEqual(error, 0.03)  # 3 % tolerance

    def test_determinism(self) -> None:
        # Same input sequence must always give the same estimate
        seq = [random.randint(0, 1_000_000) for _ in range(10_000)]
        h1 = HyperLogLog(p=13)
        h2 = HyperLogLog(p=13)
        h1.update(seq)
        h2.update(seq)
        self.assertEqual(h1.count(), h2.count())

    def test_invalid_precision(self) -> None:
        with self.assertRaises(ValueError):
            HyperLogLog(p=3)
        with self.assertRaises(ValueError):
            HyperLogLog(p=17)


def demo() -> None:
    """Simple interactive demonstration."""
    h = HyperLogLog(p=14)
    for i in range(1, 21):
        h.add(i)
        print(f"After inserting {i} elements, estimate = {h.count()}")
    print(f"Final estimate for 20 distinct elements: {h.count()}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a brief demo
    print("\n--- HyperLogLog Demo ---")
    demo()
