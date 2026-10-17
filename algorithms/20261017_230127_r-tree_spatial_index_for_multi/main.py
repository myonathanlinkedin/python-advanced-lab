import unittest
import random
from typing import Tuple, List
from core import RTree, Rect

def make_rect(dim: int, low: float, high: float) -> Rect:
    mins = tuple(random.uniform(low, high - 1) for _ in range(dim))
    maxs = tuple(m + random.uniform(0.1, 1.0) for m in mins)
    return (mins, maxs)

class TestRTree(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)

    def test_basic_insert_search(self):
        tree = RTree(max_entries=4)
        rects = [
            ((0.0, 0.0), (1.0, 1.0)),
            ((2.0, 2.0), (3.0, 3.0)),
            ((0.5, 0.5), (2.5, 2.5)),
        ]
        for i, r in enumerate(rects):
            tree.insert(r, f"obj{i}")
        # Query that overlaps first two
        q = ((0.0, 0.0), (2.0, 2.0))
        hits = set(tree.search(q))
        self.assertEqual(hits, {"obj0", "obj2"})
        # Query that matches nothing
        q2 = ((3.5, 3.5), (4.0, 4.0))
        self.assertEqual(tree.search(q2), [])

    def test_universal_query_returns_all(self):
        tree = RTree(max_entries=5)
        objs = []
        for i in range(20):
            r = make_rect(2, 0, 100)
            tree.insert(r, i)
            objs.append(i)
        universal = ((-1e9, -1e9), (1e9, 1e9))
        result = set(tree.search(universal))
        self.assertEqual(set(objs), result)

    def test_random_queries_consistency(self):
        tree = RTree(max_entries=6)
        data: List[Tuple[Rect, int]] = []
        for i in range(100):
            r = make_rect(3, 0, 50)
            tree.insert(r, i)
            data.append((r, i))
        for _ in range(30):
            q = make_rect(3, 0, 50)
            expected = {obj for rect, obj in data if
                        (rect[0][0] <= q[1][0] and rect[1][0] >= q[0][0]) and
                        (rect[0][1] <= q[1][1] and rect[1][1] >= q[0][1]) and
                        (rect[0][2] <= q[1][2] and rect[1][2] >= q[0][2])}
            self.assertEqual(set(tree.search(q)), expected)

    def test_split_and_height_growth(self):
        tree = RTree(max_entries=3)
        for i in range(15):
            r = ((i, i), (i + 0.5, i + 0.5))
            tree.insert(r, i)
        # Height should be >1 after many splits
        self.assertTrue(tree.root.leaf is False)
        # All objects must be searchable
        for i in range(15):
            q = ((i, i), (i + 0.5, i + 0.5))
            self.assertIn(i, tree.search(q))

if __name__ == '__main__':
    unittest.main()
