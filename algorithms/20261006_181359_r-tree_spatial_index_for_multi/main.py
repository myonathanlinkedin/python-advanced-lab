import unittest
import time
import random
from core import BBox, RTree


class TestBBox(unittest.TestCase):
    """Unit tests for BBox class."""

    def test_area(self):
        bbox = BBox(0, 0, 2, 3)
        self.assertEqual(bbox.area(), 6.0)

    def test_intersects_true(self):
        b1 = BBox(0, 0, 2, 2)
        b2 = BBox(1, 1, 3, 3)
        self.assertTrue(b1.intersects(b2))

    def test_intersects_false(self):
        b1 = BBox(0, 0, 1, 1)
        b2 = BBox(2, 2, 3, 3)
        self.assertFalse(b1.intersects(b2))

    def test_contains(self):
        outer = BBox(0, 0, 10, 10)
        inner = BBox(2, 2, 5, 5)
        self.assertTrue(outer.contains(inner))
        self.assertFalse(inner.contains(outer))

    def test_union(self):
        b1 = BBox(0, 0, 2, 2)
        b2 = BBox(1, 1, 3, 3)
        union = b1.union(b2)
        self.assertEqual(union.min_x, 0)
        self.assertEqual(union.min_y, 0)
        self.assertEqual(union.max_x, 3)
        self.assertEqual(union.max_y, 3)

    def test_distance_to(self):
        b1 = BBox(0, 0, 1, 1)
        b2 = BBox(3, 3, 4, 4)
        dist = b1.distance_to(b2)
        self.assertAlmostEqual(dist, math.sqrt(8.0))


class TestRTree(unittest.TestCase):
    """Unit tests for RTree class."""

    def setUp(self):
        self.tree = RTree(max_children=4, min_children=2)

    def test_insert_and_len(self):
        self.assertEqual(len(self.tree), 0)
        self.tree.insert(BBox(0, 0, 1, 1))
        self.assertEqual(len(self.tree), 1)
        self.tree.insert(BBox(2, 2, 3, 3))
        self.assertEqual(len(self.tree), 2)

    def test_query_intersects(self):
        self.tree.insert(BBox(0, 0, 1, 1))
        self.tree.insert(BBox(2, 2, 3, 3))
        self.tree.insert(BBox(0.5, 0.5, 2.5, 2.5))

        results = self.tree.query(BBox(0, 0, 1, 1))
        self.assertEqual(len(results), 2)  # First and third boxes

    def test_query_no_results(self):
        self.tree.insert(BBox(0, 0, 1, 1))
        results = self.tree.query(BBox(5, 5, 6, 6))
        self.assertEqual(len(results), 0)

    def test_nearest_neighbor(self):
        self.tree.insert(BBox(0, 0, 1, 1))
        self.tree.insert(BBox(10, 10, 11, 11))

        nearest = self.tree.nearest_neighbor((0.5, 0.5))
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest.min_x, 0)

        nearest = self.tree.nearest_neighbor((10.5, 10.5))
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest.min_x, 10)

    def test_multiple_inserts(self):
        for i in range(10):
            self.tree.insert(BBox(float(i), float(i), float(i + 1), float(i + 1)))
        self.assertEqual(len(self.tree), 10)

    def test_query_all(self):
        for i in range(5):
            self.tree.insert(BBox(float(i), float(i), float(i + 1), float(i + 1)))

        results = self.tree.query(BBox(-1, -1, 10, 10))
        self.assertEqual(len(results), 5)


class TestRTreePerformance(unittest.TestCase):
    """Performance tests for RTree."""

    def test_insert_performance(self):
        tree = RTree(max_children=8)
        n = 1000
        start = time.time()
        for i in range(n):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            tree.insert(BBox(x, y, x + 1, y + 1))
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.0)  # Should complete in under 1 second

    def test_query_performance(self):
        tree = RTree(max_children=8)
        n = 1000
        for i in range(n):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            tree.insert(BBox(x, y, x + 1, y + 1))

        start = time.time()
        for _ in range(100):
            tree.query(BBox(40, 40, 60, 60))
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.0)


def run_demo():
    """Demonstrate RTree functionality."""
    print("R-Tree Spatial Index Demo")
    print("=" * 40)

    tree = RTree(max_children=4)

    # Insert some bounding boxes
    boxes = [
        BBox(0, 0, 2, 2),
        BBox(5, 5, 7, 7),
        BBox(1, 1, 3, 3),
        BBox(10, 10, 12, 12),
        BBox(6, 6, 8, 8),
    ]

    for box in boxes:
        tree.insert(box)

    print(f"Inserted {len(tree)} bounding boxes")
    print(f"Tree: {tree}")

    # Query
    query_box = BBox(4, 4, 6, 6)
    results = tree.query(query_box)
    print(f"\nQuery {query_box}:")
    for r in results:
        print(f"  Found: {r}")

    # Nearest neighbor
    point = (5.5, 5.5)
    nearest = tree.nearest_neighbor(point)
    print(f"\nNearest to {point}: {nearest}")


if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run demo
    print("\n" + "=" * 40)
    run_demo()
