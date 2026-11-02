import unittest
import random
from core import RedBlackTree


class TestRedBlackTree(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = RedBlackTree()

    def test_insert_and_search(self) -> None:
        data = {i: str(i) for i in range(10)}
        for k, v in data.items():
            self.tree.insert(k, v)
        for k, v in data.items():
            self.assertIn(k, self.tree)
            self.assertEqual(self.tree.search(k), v)

    def test_inorder_sorted(self) -> None:
        keys = random.sample(range(1000), 200)
        for k in keys:
            self.tree.insert(k, k * 2)
        inorder_keys = [k for k, _ in self.tree.inorder()]
        self.assertEqual(inorder_keys, sorted(keys))

    def test_duplicate_insertion(self) -> None:
        self.tree.insert(5, "a")
        with self.assertRaises(ValueError):
            self.tree.insert(5, "b")

    def test_delete(self) -> None:
        keys = list(range(1, 21))
        random.shuffle(keys)
        for k in keys:
            self.tree.insert(k, k)
        for k in range(1, 21):
            self.assertIn(k, self.tree)
            self.tree.delete(k)
            self.assertNotIn(k, self.tree)
        self.assertEqual(len(self.tree), 0)

    def test_random_operations(self) -> None:
        ops = 500
        reference = {}
        for _ in range(ops):
            if random.random() < 0.6:
                k = random.randint(0, 1000)
                v = random.randint(-1000, 1000)
                if k not in reference:
                    self.tree.insert(k, v)
                    reference[k] = v
            else:
                if reference:
                    k = random.choice(list(reference.keys()))
                    self.tree.delete(k)
                    del reference[k]
        # final verification
        self.assertEqual(len(self.tree), len(reference))
        for k, v in reference.items():
            self.assertEqual(self.tree.search(k), v)
        # ensure tree invariants hold
        self.tree._assert_properties()

    def test_black_height_consistency(self) -> None:
        # Insert many keys to increase depth
        for i in range(1, 1001):
            self.tree.insert(i, i)
        # The internal assertion already checks black-height; just call it
        self.tree._assert_properties()


if __name__ == "__main__":
    unittest.main()
