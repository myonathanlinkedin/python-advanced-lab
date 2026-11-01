import unittest
import random
from typing import List
from core import ConsistentHashRing


class TestConsistentHashRing(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)  # deterministic randomness for reproducibility

    def test_empty_ring_returns_none(self) -> None:
        ring = ConsistentHashRing()
        self.assertIsNone(ring.get_node("any-key"))

    def test_single_node_all_keys_map_to_it(self) -> None:
        ring = ConsistentHashRing(nodes=["node-A"], replicas=10)
        for key in ["alpha", "beta", "gamma", "delta"]:
            self.assertEqual(ring.get_node(key), "node-A")

    def test_multiple_nodes_distribution(self) -> None:
        nodes = ["node-1", "node-2", "node-3"]
        ring = ConsistentHashRing(nodes=nodes, replicas=20)
        keys = [f"key-{i}" for i in range(300)]
        distribution = ring.get_distribution(keys)

        # Every node must appear at least once.
        for n in nodes:
            self.assertIn(n, distribution)

        # Roughly even distribution (tolerance 20%).
        expected = len(keys) / len(nodes)
        for n_keys in distribution.values():
            self.assertTrue(abs(len(n_keys) - expected) <= 0.2 * expected)

    def test_remove_node_redistributes_keys(self) -> None:
        nodes = ["A", "B", "C"]
        ring = ConsistentHashRing(nodes=nodes, replicas=30)
        keys = [f"k{i}" for i in range(200)]
        before = ring.get_distribution(keys)

        ring.remove_node("B")
        after = ring.get_distribution(keys)

        # Node B should disappear.
        self.assertNotIn("B", after)

        # Keys previously on B must now be on A or C.
        moved_keys = set(before["B"])
        self.assertTrue(moved_keys.issubset(set(after["A"] + after["C"])))

        # No key should be lost.
        total_before = sum(len(v) for v in before.values())
        total_after = sum(len(v) for v in after.values())
        self.assertEqual(total_before, total_after)

    def test_duplicate_node_addition_is_idempotent(self) -> None:
        ring = ConsistentHashRing(replicas=5)
        ring.add_node("dup")
        size_before = len(ring)
        ring.add_node("dup")  # second addition should be ignored
        self.assertEqual(size_before, len(ring))

    def test_hash_consistency_across_instances(self) -> None:
        # Two rings with identical configuration must map a key to the same node.
        nodes = ["X", "Y", "Z"]
        ring1 = ConsistentHashRing(nodes=nodes, replicas=15)
        ring2 = ConsistentHashRing(nodes=nodes, replicas=15)
        for key in ["foo", "bar", "baz", "qux"]:
            self.assertEqual(ring1.get_node(key), ring2.get_node(key))

    def test_large_number_of_virtual_nodes(self) -> None:
        ring = ConsistentHashRing(nodes=["big"], replicas=1000)
        # Even with many virtual nodes the ring should still resolve keys.
        for i in range(10):
            self.assertEqual(ring.get_node(f"key-{i}"), "big")

    def test_invalid_replica_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            ConsistentHashRing(replicas=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
