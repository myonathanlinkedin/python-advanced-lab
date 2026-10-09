import unittest
import random
from core import RTree, BoundingBox, bbox_union

class TestRTree(unittest.TestCase):
    def setUp(self):
        self.tree = RTree(max_entries=4)

    def test_insert_and_search_2d(self):
        # Insert 10 random 2D boxes
        boxes = []
        for _ in range(10):
            x1, y1 = random.uniform(0, 50), random.uniform(0, 50)
            x2, y2 = x1 + random.uniform(1, 10), y1 + random.uniform(1, 10)
            bbox = ((x1, y1), (x2, y2))
            data = f"box_{x1:.1f}_{y1:.1f}"
            boxes.append((bbox, data))
            self.tree.insert(bbox, data)
        # Query a region that should intersect some boxes
        query = ((20, 20), (30, 30))
        results = self.tree.search(query)
        # Verify that all returned boxes actually intersect the query
        for res in results:
            found = False
            for bbox, data in boxes:
                if data == res:
                    self.assertTrue(bbox_intersects(bbox, query))
                    found = True
                    break
            self.assertTrue(found)

    def test_leaf_split(self):
        # Insert enough entries to force a split
        for i in range(5):
            bbox = ((i, i), (i+1, i+1))
            self.tree.insert(bbox, f"data_{i}")
        # Root should now have two children
        self.assertFalse(self.tree.root.is_leaf)
        self.assertEqual(len(self.tree.root.entries), 2)

    def test_query_no_results(self):
        self.tree.insert(((0, 0), (5, 5)), "a")
        self.tree.insert(((10, 10), (15, 15)), "b")
        results = self.tree.search(((20, 20), (25, 25)))
        self.assertEqual(results, [])

def bbox_intersects(b1: BoundingBox, b2: BoundingBox) -> bool:
    return all(a <= d and c <= b for a, b, c, d in zip(b1[0], b1[1], b2[0], b2[1]))

if __name__ == "__main__":
    # Demo: build a tree and perform a query
    tree = RTree(max_entries=4)
    for i in range(20):
        x1, y1 = random.uniform(0, 100), random.uniform(0, 100)
        x2, y2 = x1 + random.uniform(5, 15), y1 + random.uniform(5, 15)
        bbox = ((x1, y1), (x2, y2))
        tree.insert(bbox, f"rect_{i}")
    query_box = ((30, 30), (70, 70))
    found = tree.search(query_box)
    print(f"Found {len(found)} entries intersecting {query_box}")
    unittest.main(exit=False)
