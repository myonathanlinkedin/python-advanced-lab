import unittest
from typing import List, Tuple, Set

# Import from core
from core import Graph, Demand, compute_buy_at_bulk_solution

class TestBuyAtBulk(unittest.TestCase):
    def setUp(self) -> None:
        # Build a tiny graph:
        # 0 --1-- 1 --1-- 2
        # Edge bundles: (1 unit, $5) and (5 units, $20) -> cheaper per unit for larger bundle
        self.g = Graph()
        self.g.add_edge(0, 1, length=1.0, bundles=[(1, 5.0), (5, 20.0)])
        self.g.add_edge(1, 2, length=1.0, bundles=[(1, 5.0), (5, 20.0)])

    def test_edge_cost(self):
        e = next(iter(self.g.edges()))
        # Flow 0 -> cost 0
        self.assertAlmostEqual(e.cost(0), 0.0)
        # Flow 1 -> cheapest is one 1‑unit bundle: $5
        self.assertAlmostEqual(e.cost(1), 5.0)
        # Flow 3 -> three 1‑unit bundles cheaper than one 5‑unit bundle
        self.assertAlmostEqual(e.cost(3), 15.0)
        # Flow 5 -> one 5‑unit bundle $20 cheaper than five 1‑unit bundles $25
        self.assertAlmostEqual(e.cost(5), 20.0)
        # Flow 6 -> one 5‑unit bundle + one 1‑unit bundle = $25
        self.assertAlmostEqual(e.cost(6), 25.0)

    def test_shortest_path(self):
        path = self.g.shortest_path(0, 2)
        # Should traverse both edges (ids 0 and 1 in insertion order)
        self.assertEqual(len(path), 2)
        # Verify that the path indeed connects the nodes
        # Reconstruct nodes from edges
        nodes = [0]
        for eid in path:
            edge = self.g.edge_by_id(eid)
            nxt = edge.other(nodes[-1])
            nodes.append(nxt)
        self.assertEqual(nodes, [0, 1, 2])

    def test_compute_solution(self):
        demands: List[Demand] = [Demand(src=0, dst=2, amount=3)]
        used_edges, total_cost = compute_buy_at_bulk_solution(self.g, demands)
        # Both edges should be used
        self.assertEqual(used_edges, {0, 1})
        # Each edge carries flow 3, cost per edge = 15 (see test_edge_cost)
        self.assertAlmostEqual(total_cost, 30.0)

    def test_multiple_demands(self):
        demands: List[Demand] = [
            Demand(src=0, dst=1, amount=2),
            Demand(src=1, dst=2, amount=4),
            Demand(src=0, dst=2, amount=1),
        ]
        used_edges, total_cost = compute_buy_at_bulk_solution(self.g, demands)
        # All edges are used
        self.assertEqual(used_edges, {0, 1})
        # Edge 0 carries flow 2 (0→1) + 1 (0→2) = 3 -> cost 15
        # Edge 1 carries flow 4 (1→2) + 1 (0→2) = 5 -> cost 20
        self.assertAlmostEqual(total_cost, 35.0)

    def test_no_path(self):
        # Add an isolated node 3
        self.g.add_edge(3, 3, length=0.0, bundles=[(1, 1.0)])  # self‑loop, irrelevant
        demands: List[Demand] = [Demand(src=0, dst=3, amount=1)]
        with self.assertRaises(ValueError):
            compute_buy_at_bulk_solution(self.g, demands)

if __name__ == '__main__':
    unittest.main()
