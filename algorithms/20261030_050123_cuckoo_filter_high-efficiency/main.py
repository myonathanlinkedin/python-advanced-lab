import unittest
import random
import string
from typing import List

from core import CuckooFilter


def _random_strings(count: int, length: int = 8) -> List[str]:
    rng = random.Random(0)  # deterministic
    chars = string.ascii_letters + string.digits
    return [
        ''.join(rng.choice(chars) for _ in range(length))
        for _ in range(count)
    ]


class TestCuckooFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = CuckooFilter(capacity=256, bucket_size=4, fingerprint_size=8)

    def test_basic_insert_lookup(self) -> None:
        items = _random_strings(100)
        for it in items:
            self.assertTrue(self.filter.insert(it), f"Failed to insert {it}")
        for it in items:
            self.assertTrue(self.filter.contains(it), f"Missing {it}")

    def test_deletion(self) -> None:
        items = _random_strings(50)
        for it in items:
            self.filter.insert(it)
        # delete half
        for it in items[:25]:
            self.assertTrue(self.filter.delete(it), f"Failed to delete {it}")
            self.assertFalse(self.filter.contains(it), f"Deleted item still present {it}")
        # remaining should still be present
        for it in items[25:]:
            self.assertTrue(self.filter.contains(it), f"Unexpected loss {it}")

    def test_false_positive_rate(self) -> None:
        n = 200
        m = 2000
        items = _random_strings(n)
        non_items = _random_strings(m, length=10)
        for it in items:
            self.filter.insert(it)

        false_positives = sum(1 for it in non_items if self.filter.contains(it))
        observed_fp = false_positives / m

        # theoretical upper bound: 2 * bucket_size / 2^f
        theoretical = 2 * self.filter.bucket_size / (1 << self.filter.fingerprint_size)
        # allow a small margin
        self.assertLessEqual(observed_fp, theoretical * 1.5,
                             f"Observed FP {observed_fp:.4f} exceeds bound {theoretical:.4f}")

    def test_full_filter_behavior(self) -> None:
        # Fill filter until insert fails
        items = _random_strings(5000)
        inserted = 0
        for it in items:
            if self.filter.insert(it):
                inserted += 1
            else:
                break
        self.assertGreater(inserted, 0, "No items inserted")
        self.assertFalse(self.filter.insert("extra_item"), "Insertion succeeded despite full filter")
        # Ensure load factor is high
        self.assertGreater(self.filter.load_factor(), 0.9)

    def test_idempotent_insert(self) -> None:
        item = "duplicate_test"
        self.assertTrue(self.filter.insert(item))
        # second insert should succeed (filter can store duplicate fingerprints)
        self.assertTrue(self.filter.insert(item))
        # deletion removes only one copy
        self.assertTrue(self.filter.delete(item))
        self.assertTrue(self.filter.contains(item))
        self.assertTrue(self.filter.delete(item))
        self.assertFalse(self.filter.contains(item))


if __name__ == '__main__':
    unittest.main(verbosity=2)
