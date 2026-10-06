import unittest
import random
import time
from typing import Tuple

from core import RTree, BBox

class TestRTree(unittest.TestCase):
    def setUp(self):
        self.tree = RTree(max_entries=4, min_entries=2)

    def test_insert_and_search_point(self):
        points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50)]
        for idx, (x, y) in enumerate(points):
            bbox: BBox = ((x, y), (x, y))
            self.tree.insert(f"point{idx}", bbox)
        # Search a region that should contain some points
        query: BBox = ((30, 30), (70, 70))
        results = self.tree.search(query)
        # Verify that all returned points are within query
        for r in results:
            idx = int(r.replace("point", ""))
            x, y = points[idx]
            self.assertTrue(30 <= x <= 70 and 30 <= y <= 70)

    def test_search_no_results(self):
        self.tree.insert("a", ((10, 10), (10, 10)))
        results = self.tree.search(((20, 20), (30, 30)))
        self.assertEqual(len(results), 0)

    def test_split_root(self):
        # Insert enough entries to force root split
        for i in range(10):
            bbox: BBox = ((i, i), (i, i))
            self.tree.insert(f"p{i}", bbox)
        self.assertFalse(self.tree.root.leaf)
        self.assertEqual(len(self.tree.root.entries), 2)

class Benchmark:
    def __init__(self, count: int = 10000):
        self.count = count
        self.tree = RTree(max_entries=8, min_entries=4)

    def run(self):
        start = time.time()
        for i in range(self.count):
            x, y = random.uniform(0, 1000), random.uniform(0, 1000)
            bbox: BBox = ((x, y), (x, y))
            self.tree.insert(f"pt{i}", bbox)
        insert_time = time.time() - start

        start = time.time()
        for _ in range(100):
            qx, qy = random.uniform(0, 1000), random.uniform(0, 1000)
            query: BBox = ((qx-5, qy-5), (qx+5, qy+5))
            self.tree.search(query)
        search_time = time.time() - start

        print(f"Inserted {self.count} points in {insert_time:.4f}s")
        print(f"Performed 100 searches in {search_time:.4f}s")

if __name__ == "__main__":
    unittest.main(exit=False)
    print("\nRunning benchmark...")
    Benchmark(5000).run()
