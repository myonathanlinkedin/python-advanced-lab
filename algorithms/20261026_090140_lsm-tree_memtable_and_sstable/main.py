import unittest
import shutil
from pathlib import Path

# Import the core implementation.
from core import LSMTree


class TestLSMTree(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path('tmp_lsm')
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.lsm = LSMTree(self.test_dir, memtable_threshold=3, level0_max=2)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_put_and_get(self) -> None:
        self.lsm.put(b'k1', b'v1')
        self.lsm.put(b'k2', b'v2')
        self.assertEqual(self.lsm.get(b'k1'), b'v1')
        self.assertEqual(self.lsm.get(b'k2'), b'v2')
        self.assertIsNone(self.lsm.get(b'k3'))

    def test_overwrite(self) -> None:
        self.lsm.put(b'k', b'v1')
        self.lsm.put(b'k', b'v2')
        self.assertEqual(self.lsm.get(b'k'), b'v2')

    def test_delete(self) -> None:
        self.lsm.put(b'k', b'v')
        self.lsm.delete(b'k')
        self.assertIsNone(self.lsm.get(b'k'))

    def test_flush(self) -> None:
        # memtable_threshold=3, so after 3 puts a flush occurs.
        self.lsm.put(b'a', b'1')
        self.lsm.put(b'b', b'2')
        self.lsm.put(b'c', b'3')  # triggers flush
        self.assertEqual(len(self.lsm.level0), 1)
        self.assertIsNone(self.lsm.mem._values)  # memtable cleared
        self.assertEqual(self.lsm.get(b'b'), b'2')

    def test_compaction(self) -> None:
        # Create enough sstables to trigger compaction (level0_max=2)
        self.lsm.put(b'x', b'1')
        self.lsm.put(b'y', b'2')
        self.lsm.put(b'z', b'3')  # flush 1
        self.lsm.put(b'x', b'4')
        self.lsm.put(b'y', b'5')
        self.lsm.put(b'w', b'6')  # flush 2 -> compaction
        # After compaction, level0 should be empty and level1 present.
        self.assertEqual(len(self.lsm.level0), 0)
        self.assertIsNotNone(self.lsm.level1)
        # Verify that the latest values survive.
        self.assertEqual(self.lsm.get(b'x'), b'4')
        self.assertEqual(self.lsm.get(b'y'), b'5')
        self.assertEqual(self.lsm.get(b'z'), b'3')
        self.assertEqual(self.lsm.get(b'w'), b'6')

    def test_compaction_with_deletions(self) -> None:
        self.lsm.put(b'k', b'v')
        self.lsm.put(b'k', b'v2')
        self.lsm.put(b'k', b'v3')  # flush 1
        self.lsm.delete(b'k')      # memtable now has tombstone
        self.lsm.put(b'a', b'1')
        self.lsm.put(b'b', b'2')
        self.lsm.put(b'c', b'3')   # flush 2 -> compaction
        self.assertIsNone(self.lsm.get(b'k'))
        self.assertEqual(self.lsm.get(b'a'), b'1')

    def test_iteration_order(self) -> None:
        # Ensure iterator yields keys in sorted order.
        self.lsm.put(b'c', b'3')
        self.lsm.put(b'a', b'1')
        self.lsm.put(b'b', b'2')
        self.lsm.put(b'd', b'4')  # triggers flush
        keys = [k for k, _ in self.lsm.level0[0].iterator()]
        self.assertEqual(keys, [b'a', b'b', b'c', b'd'])


if __name__ == '__main__':
    unittest.main()
