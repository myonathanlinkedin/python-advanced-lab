import unittest
from typing import List, Tuple
from core import RTree, Rect

class TestRTree(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = RTree(max_entries=4)

    def test_single_insert_search(self) -> None:
        r: Rect = ((1.0, 1.0), (2.0, 2.0))
        self.tree.insert(r, "A")
        self.assertEqual(len(self.tree), 1)
        hits = self.tree.search(((0.0, 0.0), (3.0, 3.0)))
        self.assertIn("A", hits)

    def test_no_overlap(self) -> None:
        self.tree.insert(((0, 0), (1, 1)), "A")
        self.tree.insert(((2, 2), (3, 3)), "B")
        hits = self.tree.search(((1.1, 1.1), (1.9, 1.9)))
        self.assertEqual(hits, [])

    def test_multiple_overlaps(self) -> None:
        objs = [("A", ((0,0),(2,2))), ("B", ((1,1),(3,3))), ("C", ((2,0),(4,2)))]
        for obj, rect in objs:
            self.tree.insert(rect, obj)
        hits = self.tree.search(((1.5,0.5),(2.5,1.5)))
        self.assertCountEqual(hits, ["A","B","C"])

    def test_tree_growth(self) -> None:
        # Insert enough rectangles to force multiple splits and height increase
        for i in range(30):
            rect = ((i, i), (i+0.5, i+0.5))
            self.tree.insert(rect, i)
        self.assertEqual(len(self.tree), 30)
        # Query a range that includes several entries
        hits = self.tree.search(((10,10),(15,15)))
        expected = list(range(10,16))
        self.assertCountEqual(hits, expected)

    def test_edge_touching(self) -> None:
        self.tree.insert(((0,0),(1,1)), "A")
        self.tree.insert(((1,1),(2,2)), "B")  # touches at corner
        hits = self.tree.search(((1,1),(1,1)))  # point query at touching corner
        self.assertCountEqual(hits, ["A","B"])

if __name__ == "__main__":
    unittest.main()
