import unittest
import random
from core import RTree, Rectangle

class TestRTree(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.tree = RTree(self.dim)

    def random_rect(self, size=10.0):
        mins = tuple(random.uniform(0, 50) for _ in range(self.dim))
        maxs = tuple(mins[i] + random.uniform(0, size) for i in range(self.dim))
        return Rectangle(mins, maxs)

    def test_insert_and_search(self):
        data = []
        for _ in range(50):
            r = self.random_rect()
            d = f"obj_{_}"
            self.tree.insert(r, d)
            data.append((r, d))
        # Pick a random query rectangle
        q = self.random_rect()
        expected = [d for r, d in data if r.intersects(q)]
        result = self.tree.search(q)
        self.assertCountEqual(expected, result)

    def test_leaf_split(self):
        # Insert enough entries to force a split
        for i in range(10):
            r = Rectangle((i, i), (i+1, i+1))
            self.tree.insert(r, f"leaf_{i}")
        # Root should be internal after split
        self.assertFalse(self.tree.root.is_leaf)
        # Search for a point
        q = Rectangle((5,5),(5,5))
        res = self.tree.search(q)
        self.assertIn("leaf_5", res)

    def test_internal_split(self):
        # Build a tree that splits internal nodes
        for i in range(30):
            r = Rectangle((i, i), (i+1, i+1))
            self.tree.insert(r, f"obj_{i}")
        # Root should be internal
        self.assertFalse(self.tree.root.is_leaf)
        # Search for a range
        q = Rectangle((10,10),(20,20))
        res = self.tree.search(q)
        expected = [f"obj_{i}" for i in range(10,20)]
        self.assertCountEqual(expected, res)

if __name__ == "__main__":
    # Demo
    tree = RTree(2)
    for i in range(5):
        rect = Rectangle((i*10, i*10), ((i+1)*10, (i+1)*10))
        tree.insert(rect, f"box_{i}")
    query = Rectangle((15,15),(35,35))
    print("Search results:", tree.search(query))
    # Run tests
    unittest.main(exit=False)
