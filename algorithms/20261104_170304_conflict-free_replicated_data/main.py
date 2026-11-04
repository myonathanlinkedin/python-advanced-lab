import unittest
from typing import List
from core import PNCounter


class TestPNCounter(unittest.TestCase):
    def setUp(self) -> None:
        self.replica_a = "A"
        self.replica_b = "B"

    def test_initial_value_is_zero(self) -> None:
        c = PNCounter()
        self.assertEqual(c.value(), 0)

    def test_increment_and_decrement(self) -> None:
        c = PNCounter()
        c.increment(self.replica_a, 5)
        c.decrement(self.replica_a, 2)
        self.assertEqual(c.value(), 3)

    def test_multiple_replicas(self) -> None:
        c = PNCounter()
        c.increment(self.replica_a, 3)
        c.increment(self.replica_b, 4)
        c.decrement(self.replica_a, 1)
        c.decrement(self.replica_b, 2)
        self.assertEqual(c.value(), (3 + 4) - (1 + 2))

    def test_merge_commutative(self) -> None:
        c1 = PNCounter()
        c2 = PNCounter()
        c1.increment(self.replica_a, 2)
        c2.increment(self.replica_b, 3)
        c1.decrement(self.replica_a, 1)
        c2.decrement(self.replica_b, 1)

        merged_ab = PNCounter()
        merged_ab.merge(c1)
        merged_ab.merge(c2)

        merged_ba = PNCounter()
        merged_ba.merge(c2)
        merged_ba.merge(c1)

        self.assertEqual(merged_ab.value(), merged_ba.value())
        self.assertEqual(merged_ab, merged_ba)

    def test_merge_associative(self) -> None:
        c1 = PNCounter()
        c2 = PNCounter()
        c3 = PNCounter()
        c1.increment(self.replica_a, 1)
        c2.increment(self.replica_b, 2)
        c3.increment(self.replica_a, 3)
        c3.decrement(self.replica_b, 1)

        # (c1 merge c2) merge c3
        left = PNCounter()
        left.merge(c1)
        left.merge(c2)
        left.merge(c3)

        # c1 merge (c2 merge c3)
        right = PNCounter()
        right.merge(c1)
        tmp = PNCounter()
        tmp.merge(c2)
        tmp.merge(c3)
        right.merge(tmp)

        self.assertEqual(left.value(), right.value())
        self.assertEqual(left, right)

    def test_merge_idempotent(self) -> None:
        c = PNCounter()
        c.increment(self.replica_a, 4)
        c.decrement(self.replica_a, 2)

        merged = PNCounter()
        merged.merge(c)
        merged.merge(c)  # second merge should not change state

        self.assertEqual(merged.value(), c.value())
        self.assertEqual(merged, c)

    def test_negative_increment_error(self) -> None:
        c = PNCounter()
        with self.assertRaises(ValueError):
            c.increment(self.replica_a, -1)

    def test_negative_decrement_error(self) -> None:
        c = PNCounter()
        with self.assertRaises(ValueError):
            c.decrement(self.replica_a, 0)

    def test_serialisation_roundtrip(self) -> None:
        c = PNCounter()
        c.increment(self.replica_a, 7)
        c.decrement(self.replica_b, 3)
        data = c.to_dict()
        restored = PNCounter.from_dict(data)
        self.assertEqual(c, restored)
        self.assertEqual(c.value(), restored.value())

    def test_large_numbers(self) -> None:
        c = PNCounter()
        big = 10**12
        c.increment(self.replica_a, big)
        c.decrement(self.replica_a, big - 1)
        self.assertEqual(c.value(), 1)


if __name__ == "__main__":
    unittest.main()
