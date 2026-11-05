import unittest
import random
from collections import Counter
from core import ConsistentHashRing


class TestConsistentHashRing(unittest.TestCase):
    def setUp(self) -> None:
        self.ring = ConsistentHashRing(replicas=10)

    def test_add_and_remove_node(self):
        self.ring.add_node("nodeA")
        self.assertIn("nodeA", self.ring.nodes())
        self.ring.add_node("nodeB")
        self.assertCountEqual(self.ring.nodes(), ["nodeA", "nodeB"])
        # Removing
        self.ring.remove_node("nodeA")
        self.assertNotIn("nodeA", self.ring.nodes())
        self.assertIn("nodeB", self.ring.nodes())
        # Removing non‑existent node raises
        with self.assertRaises(KeyError):
            self.ring.remove_node("nodeX")

    def test_duplicate_node(self):
        self.ring.add_node("nodeA")
        with self.assertRaises(ValueError):
            self.ring.add_node("nodeA")

    def test_get_node_basic(self):
        nodes = ["A", "B", "C"]
        for n in nodes:
            self.ring.add_node(n)
        for key in ["alpha", "beta", "gamma", "delta"]:
            node = self.ring.get_node(key)
            self.assertIn(node, nodes)

    def test_get_node_empty_ring(self):
        empty_ring = ConsistentHashRing()
        with self.assertRaises(RuntimeError):
            empty_ring.get_node("anything")

    def test_distribution_uniformity(self):
        # Add 5 nodes, hash 10,000 random keys, expect roughly uniform distribution
        for i in range(5):
            self.ring.add_node(f"node{i}")
        keys = [f"key{j}" for j in range(10000)]
        assignments = [self.ring.get_node(k) for k in keys]
        counts = Counter(assignments)
        # Each node should have between 15% and 25% of keys (allowing variance)
        for cnt in counts.values():
            self.assertGreater(cnt, 0.15 * len(keys))
            self.assertLess(cnt, 0.25 * len(keys))

    def test_get_nodes_replication(self):
        for i in range(3):
            self.ring.add_node(f"N{i}")
        key = "important-data"
        nodes = self.ring.get_nodes(key, 2)
        self.assertEqual(len(nodes), 2)
        self.assertNotEqual(nodes[0], nodes[1])
        # Request more than available nodes returns all distinct nodes
        all_nodes = self.ring.get_nodes(key, 10)
        self.assertCountEqual(all_nodes, self.ring.nodes())

    def test_invalid_replica_count(self):
        with self.assertRaises(ValueError):
            ConsistentHashRing(replicas=0)

    def test_invalid_get_nodes_count(self):
        self.ring.add_node("X")
        with self.assertRaises(ValueError):
            self.ring.get_nodes("k", 0)


def demo():
    """Simple demonstration of key routing."""
    ring = ConsistentHashRing(replicas=5)
    for node in ["us-east", "us-west", "eu-central"]:
        ring.add_node(node)
    sample_keys = ["user:123", "order:456", "session:789"]
    for k in sample_keys:
        print(f"Key {k!r} → Node {ring.get_node(k)!r}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n--- Demo ---")
    demo()
