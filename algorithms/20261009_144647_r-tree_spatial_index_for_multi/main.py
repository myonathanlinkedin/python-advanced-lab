import unittest
from typing import List, Tuple
from core import RTree, BBox

class TestRTree(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = RTree(max_entries=4)  # small M to force splits

    def _make_box(self, x: float, y: float, w: float = 1.0, h: float = 1.0) -> BBox:
        return ((x, y), (x + w, y + h))

    def test_single_insert_and_search(self) -> None:
        box = self._make_box(0, 0)
        self.tree.insert(box, "A")
        self.assertListEqual(self.tree.search(box), ["A"])
        self.assertListEqual(self.tree.search(((5,5),(6,6))), [])

    def test_multiple_inserts_and_overlap(self) -> None:
        data = [
            (self._make_box(0, 0), "A"),
            (self._make_box(2, 2), "B"),
            (self._make_box(4, 4), "C"),
            (self._make_box(6, 6), "D"),
        ]
        for b, o in data:
            self.tree.insert(b, o)

        # Query that overlaps A and B
        query = ((1,1),(3,3))
        result = sorted(self.tree.search(query))
        self.assertEqual(result, ["A", "B"])

        # Query that overlaps none
        self.assertListEqual(self.tree.search(((10,10),(12,12))), [])

    def test_split_and_search(self) -> None:
        # Insert more than max_entries to trigger node splits
        for i in range(10):
            self.tree.insert(self._make_box(i, i), f"obj{i}")

        # All objects should be searchable
        for i in range(10):
            q = self._make_box(i, i)
            self.assertIn(f"obj{i}", self.tree.search(q))

        # Overlap query covering half the objects
        query = ((3,3),(7,7))
        result = sorted(self.tree.search(query))
        expected = [f"obj{i}" for i in range(3,8)]
        self.assertEqual(result, expected)

    def test_edge_case_zero_area_box(self) -> None:
        # Zero‑area boxes (point) should be handled
        point = ((5.0,5.0),(5.0,5.0))
        self.tree.insert(point, "P")
        self.assertIn("P", self.tree.search(((4,4),(6,6))))
        self.assertIn("P", self.tree.search(point))

    def test_degenerate_dimensions(self) -> None:
        # 3‑D boxes
        tree3d = RTree(max_entries=3)
        box3d = ((0.0,0.0,0.0),(1.0,1.0,1.0))
        tree3d.insert(box3d, "cube")
        self.assertIn("cube", tree3d.search(((0.5,0.5,0.5),(2,2,2))))
        self.assertNotIn("cube", tree3d.search(((2,2,2),(3,3,3))))

if __name__ == "__main__":
    unittest.main()
