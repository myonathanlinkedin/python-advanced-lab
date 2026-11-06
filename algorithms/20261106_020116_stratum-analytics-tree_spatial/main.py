import unittest
from typing import Tuple, List
from core import StratumAnalyticsTree, BoxTuple

def make_box(mins: Tuple[float, ...], maxs: Tuple[float, ...]) -> BoxTuple:
    return (mins, maxs)

class TestStratumAnalyticsTree(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = StratumAnalyticsTree(max_entries=4)

    def test_single_insert_and_query(self):
        b = make_box((0.0, 0.0), (1.0, 1.0))
        self.tree.insert(b, "A")
        res = self.tree.search(b)
        self.assertIn("A", res)
        self.assertEqual(len(res), 1)

    def test_non_overlapping_query(self):
        self.tree.insert(make_box((0, 0), (1, 1)), "A")
        self.tree.insert(make_box((2, 2), (3, 3)), "B")
        res = self.tree.search(make_box((1.1, 1.1), (1.9, 1.9)))
        self.assertEqual(res, [])

    def test_multiple_dimensions(self):
        dim = 5
        mins = tuple(0.0 for _ in range(dim))
        maxs = tuple(1.0 for _ in range(dim))
        self.tree.insert(make_box(mins, maxs), "high_dim")
        query = make_box(tuple(0.5 for _ in range(dim)), tuple(2.0 for _ in range(dim)))
        self.assertIn("high_dim", self.tree.search(query))

    def test_overlap_multiple_objects(self):
        self.tree.insert(make_box((0, 0), (5, 5)), "big")
        self.tree.insert(make_box((1, 1), (2, 2)), "small1")
        self.tree.insert(make_box((3, 3), (4, 4)), "small2")
        res = self.tree.search(make_box((1.5, 1.5), (3.5, 3.5)))
        self.assertCountEqual(res, ["big", "small1", "small2"])

    def test_split_and_balance(self):
        # Insert enough boxes to force several splits
        for i in range(20):
            b = make_box((i, i), (i + 0.5, i + 0.5))
            self.tree.insert(b, f"obj{i}")
        # Query a range that should intersect a known subset
        res = self.tree.search(make_box((5, 5), (7, 7)))
        expected = [f"obj{i}" for i in range(5, 8)]
        self.assertCountEqual(res, expected)

    def test_zero_volume_box(self):
        b = make_box((1, 1), (1, 1))
        self.tree.insert(b, "point")
        self.assertIn("point", self.tree.search(b))
        self.assertIn("point", self.tree.search(make_box((0, 0), (2, 2))))

    def test_invalid_max_entries(self):
        with self.assertRaises(ValueError):
            StratumAnalyticsTree(max_entries=2)

if __name__ == "__main__":
    unittest.main()
