import random
import unittest
from core import RTree, Rect, rect_combine, rect_intersect


def make_rect(dim: int, low: float = 0.0, high: float = 100.0) -> Rect:
    mins = tuple(random.uniform(low, high - 1) for _ in range(dim))
    maxs = tuple(mins[i] + random.uniform(0.1, 5.0) for i in range(dim))
    return mins, maxs


class TestRTree(unittest.TestCase):
    def test_basic_insert_search(self):
        tree = RTree(max_entries=4)
        a = ((0.0, 0.0), (2.0, 2.0))
        b = ((1.0, 1.0), (3.0, 3.0))
        c = ((3.0, 3.0), (5.0, 5.0))
        tree.insert(a, "A")
        tree.insert(b, "B")
        tree.insert(c, "C")
        self.assertCountEqual(tree.search(((1.5, 1.5), (4.0, 4.0))), ["B", "C"])
        self.assertCountEqual(tree.search(((0.0, 0.0), (1.0, 1.0))), ["A"])

    def test_no_overlap(self):
        tree = RTree()
        rects = [((i, i), (i + 0.5, i + 0.5)) for i in range(5)]
        for i, r in enumerate(rects):
            tree.insert(r, i)
        self.assertEqual(tree.search(((10, 10), (12, 12))), [])

    def test_bulk_random(self):
        dim = 3
        tree = RTree(max_entries=8)
        objs = {}
        for i in range(200):
            r = make_rect(dim)
            objs[i] = r
            tree.insert(r, i)
        # query each inserted rectangle; should contain its own id
        for i, r in objs.items():
            hits = tree.search(r)
            self.assertIn(i, hits)

    def test_root_split(self):
        tree = RTree(max_entries=3)
        for i in range(10):
            tree.insert(((i, i), (i + 1, i + 1)), i)
        # After many inserts root should be internal
        self.assertFalse(tree.root.leaf)
        # Height should be at least 2
        self.assertTrue(any(not child.leaf for _, child in tree.root.entries))

    def test_touching_edges(self):
        tree = RTree()
        a = ((0, 0), (2, 2))
        b = ((2, 2), (4, 4))  # touches a at corner
        tree.insert(a, "A")
        tree.insert(b, "B")
        self.assertCountEqual(tree.search(((2, 2), (2, 2))), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
