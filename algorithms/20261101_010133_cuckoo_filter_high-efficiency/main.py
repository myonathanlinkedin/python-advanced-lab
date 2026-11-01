import unittest
from core import CuckooFilter


class TestCuckooFilter(unittest.TestCase):
    def test_basic_operations(self):
        cf = CuckooFilter(capacity=10)
        keys = [f"key{i}".encode() for i in range(5)]
        for k in keys:
            self.assertTrue(cf.insert(k))
        for k in keys:
            self.assertTrue(cf.contains(k))
        for k in keys:
            self.assertTrue(cf.delete(k))
        for k in keys:
            self.assertFalse(cf.contains(k))

    def test_over_capacity(self):
        cf = CuckooFilter(capacity=5, bucket_size=2, max_kicks=10)
        keys = [f"k{i}".encode() for i in range(10)]
        inserted = 0
        for k in keys:
            if cf.insert(k):
                inserted += 1
        self.assertLessEqual(inserted, 10)
        # All inserted keys should be present
        for k in keys[:inserted]:
            self.assertTrue(cf.contains(k))
        # Non-inserted keys should not be present
        for k in keys[inserted:]:
            self.assertFalse(cf.contains(k))

    def test_delete_nonexistent(self):
        cf = CuckooFilter(capacity=5)
        self.assertFalse(cf.delete(b"missing"))

    def test_fingerprint_zero_reservation(self):
        cf = CuckooFilter(capacity=5, fingerprint_size=1)
        key = b"fp0"
        fp = cf._fingerprint(key)
        self.assertNotEqual(fp, 0)

    def test_bucket_count_power_of_two(self):
        cf = CuckooFilter(capacity=13)
        self.assertTrue((cf.bucket_count & (cf.bucket_count - 1)) == 0)

    def test_max_kicks_limit(self):
        cf = CuckooFilter(capacity=2, bucket_size=1, max_kicks=1)
        self.assertTrue(cf.insert(b"a"))
        self.assertTrue(cf.insert(b"b"))
        # Third insertion should fail due to max_kicks=1
        self.assertFalse(cf.insert(b"c"))

    def test_zero_capacity_error(self):
        with self.assertRaises(ValueError):
            CuckooFilter(capacity=0)

    def test_invalid_fingerprint_size(self):
        with self.assertRaises(ValueError):
            CuckooFilter(capacity=10, fingerprint_size=0)
        with self.assertRaises(ValueError):
            CuckooFilter(capacity=10, fingerprint_size=33)

    def test_repr(self):
        cf = CuckooFilter(capacity=8, bucket_size=4, fingerprint_size=16)
        rep = repr(cf)
        self.assertIn("CuckooFilter", rep)
        self.assertIn("bucket_size=4", rep)
        self.assertIn("fingerprint_size=16", rep)


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demo
    print("\nDemo: Cuckoo Filter")
    demo_cf = CuckooFilter(capacity=20)
    demo_keys = [f"demo{i}".encode() for i in range(15)]
    for k in demo_keys:
        demo_cf.insert(k)
    for k in demo_keys:
        print(f"{k.decode()}: {'present' if demo_cf.contains(k) else 'absent'}")
    print("\nDeleting 'demo5'")
    demo_cf.delete(b"demo5")
    print(f"demo5: {'present' if demo_cf.contains(b'demo5') else 'absent'}")
