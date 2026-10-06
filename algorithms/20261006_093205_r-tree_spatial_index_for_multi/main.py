import unittest
import random
from typing import Tuple
from core import RTree, Box

def make_random_box(dim: int, low: float = 0.0, high: float = 100.0, size: float = 5.0) -> Box:
    mins = tuple(random.uniform(low, high - size) for _ in range(dim))
    maxs = tuple(mins[i] + random.uniform(0.1, size) for i in range(dim))
    return mins, maxs

class TestRTree(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.dim = 2
        self.tree = RTree(dimension=self.dim, max_entries=4)

    def test_insert_and_search_single(self):
        box = ((10.0, 10.0), (20.0, 20.0))
        obj = "A"
        self.tree.insert(box, obj)
        found = self.tree.search(((15.0, 15.0), (25.0, 25.0)))
        self.assertIn(obj, found)

    def test_search_no_overlap(self):
        box = ((0.0, 0.0), (5.0, 5.0))
        self.tree.insert(box, "B")
        result = self.tree.search(((10.0, 10.0), (15.0, 15.0)))
        self.assertEqual(len(result), 0)

    def test_multiple_inserts_and_depth(self):
        boxes_objs = [(make_random_box(self.dim), i) for i in range(20)]
        for b, o in boxes_objs:
            self.tree.insert(b, o)
        # Ensure tree depth > 1 (i.e., root is not leaf)
        self.assertFalse(self.tree.root.leaf)
        # Verify all objects are searchable
        for b, o in boxes_objs:
            query = (b[0], b[1])  # exact match
            found = self.tree.search(query)
            self.assertIn(o, found)

    def test_overlap_multiple_objects(self):
        common = ((30.0, 30.0), (40.0, 40.0))
        objs = ["X", "Y", "Z"]
        for o in objs:
            self.tree.insert(common, o)
        result = self.tree.search(((35.0, 35.0), (36.0, 36.0)))
        self.assertCountEqual(result, objs)

if __name__ == '__main__':
    unittest.main()
