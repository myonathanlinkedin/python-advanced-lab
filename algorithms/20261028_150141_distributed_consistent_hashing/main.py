import unittest
from custom_types import Node
from engine import ConsistentHashRouter

class TestConsistentHashRouter(unittest.TestCase):
    def setUp(self):
        self.router = ConsistentHashRouter(virtual_nodes_per_node=10)

    def test_add_and_get(self):
        node_a = Node("A")
        node_b = Node("B")
        self.router.add_node(node_a)
        self.router.add_node(node_b)
        # Ensure keys map to one of the nodes
        for key in ["foo", "bar", "baz", "qux"]:
            node = self.router.get_node(key)
            self.assertIn(node, [node_a, node_b])

    def test_remove(self):
        node_a = Node("A")
        node_b = Node("B")
        self.router.add_node(node_a)
        self.router.add_node(node_b)
        self.router.remove_node(node_a)
        for key in ["foo", "bar"]:
            node = self.router.get_node(key)
            self.assertEqual(node, node_b)

    def test_wrap_around(self):
        node_a = Node("A")
        self.router.add_node(node_a)
        # Use a key that hashes to a value beyond the last virtual node
        key = "wraparound_test_key"
        node = self.router.get_node(key)
        self.assertEqual(node, node_a)

    def test_empty_ring(self):
        with self.assertRaises(RuntimeError):
            self.router.get_node("anykey")

    def test_duplicate_node_addition(self):
        node_a = Node("A")
        self.router.add_node(node_a)
        with self.assertRaises(ValueError):
            self.router.add_node(node_a)

    def test_virtual_node_count(self):
        node_a = Node("A")
        self.router.add_node(node_a)
        self.assertEqual(len(self.router._ring), 10)
        self.assertEqual(len(self.router._node_to_hashes[node_a]), 10)

if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demo
    print("\n=== Demo: Consistent Hashing Router ===")
    router = ConsistentHashRouter(virtual_nodes_per_node=20)
    nodes = [Node(f"Node{i}") for i in range(3)]
    for node in nodes:
        router.add_node(node)
    sample_keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
    for key in sample_keys:
        responsible = router.get_node(key)
        print(f"Key '{key}' is routed to {responsible.id}")
