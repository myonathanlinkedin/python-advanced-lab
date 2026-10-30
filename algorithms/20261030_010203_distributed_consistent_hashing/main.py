import unittest
import random
from collections import Counter
from core import ConsistentHashRing


class TestConsistentHashRing(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)

    def test_empty_ring_error(self):
        ring = ConsistentHashRing()
        with self.assertRaises(ValueError):
            ring.get_node("anykey")

    def test_add_and_get_node(self):
        ring = ConsistentHashRing(replicas=10)
        ring.add_node("A")
        ring.add_node("B")
        ring.add_node("C")
        self.assertIn(ring.get_node("my-key"), {"A", "B", "C"})

    def test_duplicate_node(self):
        ring = ConsistentHashRing()
        ring.add_node("node1")
        with self.assertRaises(ValueError):
            ring.add_node("node1")

    def test_remove_node(self):
        ring = ConsistentHashRing()
        ring.add_node("X")
        ring.add_node("Y")
        key = "important-data"
        node_before = ring.get_node(key)
        ring.remove_node(node_before)
        node_after = ring.get_node(key)
        self.assertNotEqual(node_before, node_after)
        self.assertIn(node_after, {"X", "Y"} - {node_before})

    def test_get_nodes_multiple(self):
        ring = ConsistentHashRing()
        for nid in ["N1", "N2", "N3", "N4"]:
            ring.add_node(nid)
        nodes = ring.get_nodes("some-key", 3)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(set(nodes)), 3)  # distinct

    def test_weighted_distribution(self):
        ring = ConsistentHashRing(replicas=20)
        ring.add_node("light", weight=1)
        ring.add_node("heavy", weight=5)  # 5× virtual nodes
        counts = Counter()
        for i in range(5000):
            key = f"key-{i}"
            counts[ring.get_node(key)] += 1
        # heavy should receive roughly 5× the keys of light
        ratio = counts["heavy"] / counts["light"]
        self.assertTrue(4.0 < ratio < 6.0, f"ratio={ratio}")

    def test_balanced_distribution(self):
        ring = ConsistentHashRing(replicas=30)
        nodes = ["A", "B", "C", "D"]
        for n in nodes:
            ring.add_node(n)
        counts = Counter()
        total_keys = 8000
        for i in range(total_keys):
            counts[ring.get_node(f"key-{i}")] += 1
        # Expect each node to have ~25% ±5%
        for n in nodes:
            proportion = counts[n] / total_keys
            self.assertTrue(0.20 <= proportion <= 0.30, f"{n} proportion={proportion}")

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            ConsistentHashRing(replicas=0)
        ring = ConsistentHashRing()
        with self.assertRaises(ValueError):
            ring.add_node("bad", weight=0)
        with self.assertRaises(ValueError):
            ring.get_nodes("k", 0)

    def test_len(self):
        ring = ConsistentHashRing()
        self.assertEqual(len(ring), 0)
        ring.add_node("one")
        self.assertEqual(len(ring), 1)
        ring.add_node("two")
        self.assertEqual(len(ring), 2)
        ring.remove_node("one")
        self.assertEqual(len(ring), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
