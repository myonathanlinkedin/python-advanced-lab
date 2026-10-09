import unittest
import random
from core import RTree, BoundingBox

class TestRTree(unittest.TestCase):
    def setUp(self):
        self.tree = RTree(max_entries=4)
        self.boxes = []
        for i in range(20):
            min_pt = (random.uniform(0, 50), random.uniform(0, 50))
            max_pt = (min_pt[0] + random.uniform(1, 10), min_pt[1] + random.uniform(1, 10))
            bbox = (min_pt, max_pt)
            self.boxes.append((bbox, f"obj_{i}"))
            self.tree.insert(bbox, f"obj_{i}")

    def test_search_contains(self):
        # Pick a random box and query with itself
        bbox, data = random.choice(self.boxes)
        results = self.tree.search(bbox)
        self.assertIn(data, results)

    def test_search_overlap(self):
        # Create a query that overlaps multiple boxes
        query = ((10, 10), (30, 30))
        results = self.tree.search(query)
        # Verify that all returned objects indeed overlap
        for res in results:
            found = False
            for bbox, data in self.boxes:
                if data == res and self._bbox_overlaps(bbox, query):
                    found = True
                    break
            self.assertTrue(found, f"Result {res} does not overlap query")

    def _bbox_overlaps(self, a: BoundingBox, b: BoundingBox) -> bool:
        min_a, max_a = a
        min_b, max_b = b
        return all(ma >= mb_min and mb >= ma_min for ma_min, ma, mb_min, mb in zip(min_a, max_a, min_b, max_b))

    def test_tree_structure(self):
        # Ensure root is not leaf after many insertions
        self.assertFalse(self.tree.root.is_leaf)

if __name__ == "__main__":
    # Demo: build tree and perform a query
    demo_tree = RTree(max_entries=4)
    demo_boxes = [
        ((0, 0), (5, 5)),
        ((3, 3), (8, 8)),
        ((6, 1), (10, 4)),
        ((9, 9), (12, 12)),
    ]
    for i, bbox in enumerate(demo_boxes):
        demo_tree.insert(bbox, f"demo_{i}")

    query_box = ((4, 4), (9, 9))
    found = demo_tree.search(query_box)
    print(f"Objects overlapping {query_box}: {found}")

    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
