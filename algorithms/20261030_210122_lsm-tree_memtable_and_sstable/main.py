import unittest
from core import LSMTree

class TestLSMTree(unittest.TestCase):
    def setUp(self):
        self.tree = LSMTree(memtable_size=3, compaction_threshold=2)

    def test_put_get(self):
        self.tree.put('a', 1)
        self.tree.put('b', 2)
        self.assertEqual(self.tree.get('a'), 1)
        self.assertEqual(self.tree.get('b'), 2)
        self.assertIsNone(self.tree.get('c'))

    def test_delete(self):
        self.tree.put('a', 1)
        self.tree.delete('a')
        self.assertIsNone(self.tree.get('a'))

    def test_flush(self):
        self.tree.put('a', 1)
        self.tree.put('b', 2)
        self.tree.put('c', 3)  # triggers flush
        self.assertEqual(len(self.tree.sstables), 1)
        self.assertIsNone(self.tree.memtable.get('a'))
        self.assertEqual(self.tree.get('a'), 1)

    def test_compaction(self):
        self.tree.put('a', 1)
        self.tree.put('b', 2)
        self.tree.put('c', 3)  # flush 1
        self.tree.put('a', 4)
        self.tree.put('d', 5)
        self.tree.put('e', 6)  # flush 2 triggers compaction
        self.assertEqual(len(self.tree.sstables), 1)
        self.assertEqual(self.tree.get('a'), 4)
        self.assertEqual(self.tree.get('b'), 2)
        self.assertEqual(self.tree.get('c'), 3)
        self.assertEqual(self.tree.get('d'), 5)
        self.assertEqual(self.tree.get('e'), 6)

    def test_duplicate_keys(self):
        self.tree.put('a', 1)
        self.tree.put('a', 2)
        self.assertEqual(self.tree.get('a'), 2)

    def test_tombstone_persistence(self):
        self.tree.put('a', 1)
        self.tree.put('b', 2)
        self.tree.put('c', 3)  # flush
        self.tree.delete('b')
        self.tree.put('d', 4)
        self.tree.put('e', 5)
        self.tree.put('f', 6)  # flush triggers compaction
        self.assertIsNone(self.tree.get('b'))
        self.assertEqual(self.tree.get('a'), 1)
        self.assertEqual(self.tree.get('d'), 4)

if __name__ == '__main__':
    unittest.main()
