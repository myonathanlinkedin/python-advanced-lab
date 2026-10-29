import unittest
import random
import time
from typing import List

# Import from core
from core import (
    Graph,
    tour_cost,
    random_tour,
    tsp_neighbor_swap,
    SimulatedAnnealing,
)

class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self) -> None:
        # Deterministic RNG for reproducibility
        self.rng = random.Random(42)

    def test_graph_construction(self) -> None:
        mat = [
            [0, 2, 9],
            [2, 0, 3],
            [9, 3, 0],
        ]
        g = Graph(mat)
        self.assertEqual(g.n, 3)
        self.assertAlmostEqual(g.distance(0, 1), 2)
        self.assertAlmostEqual(g.distance(1, 2), 3)

    def test_tour_cost(self) -> None:
        mat = [
            [0, 1, 2],
            [1, 0, 4],
            [2, 4, 0],
        ]
        g = Graph(mat)
        tour = [0, 1, 2]
        self.assertAlmostEqual(tour_cost(g, tour), 1 + 4 + 2)  # 0→1,1→2,2→0

    def test_random_tour(self) -> None:
        tour = random_tour(5, self.rng)
        self.assertCountEqual(tour, list(range(5)))  # permutation check

    def test_neighbor_swap(self) -> None:
        tour = [0, 1, 2, 3]
        new = tsp_neighbor_swap(tour, self.rng)
        self.assertCountEqual(new, tour)
        self.assertNotEqual(new, tour)  # with high probability; deterministic seed ensures swap

    def test_sa_on_small_tsp(self) -> None:
        # Complete graph with 4 nodes, known optimal tour cost = 10
        mat = [
            [0, 2, 9, 10],
            [2, 0, 6, 4],
            [9, 6, 0, 8],
            [10, 4, 8, 0],
        ]
        g = Graph(mat)

        def cost(state: List[int]) -> float:
            return tour_cost(g, state)

        initial = random_tour(g.n, self.rng)

        sa = SimulatedAnnealing(
            initial_temp=100.0,
            cooling_rate=0.85,
            iteration_per_temp=200,
            min_temp=1e-4,
            rng=self.rng,
        )
        best_state, best_cost = sa.optimize(initial, cost, tsp_neighbor_swap)

        # Known optimal tours: [0,1,3,2] or its rotations/reversals, cost = 2+4+8+9=23? Wait compute.
        # Let's compute manually: best possible is 2+4+8+9 = 23? Actually we can brute force.
        # Brute‑force check to assert optimality.
        from itertools import permutations

        optimal = min(tour_cost(g, p) for p in permutations(range(g.n)))
        self.assertAlmostEqual(best_cost, optimal, places=4)

    def test_edge_cases(self) -> None:
        # Single node graph
        mat = [[0]]
        g = Graph(mat)
        tour = [0]
        self.assertAlmostEqual(tour_cost(g, tour), 0)

        # SA on single node should return the same tour
        sa = SimulatedAnnealing(rng=self.rng)
        best_state, best_cost = sa.optimize(tour, lambda s: tour_cost(g, s), tsp_neighbor_swap)
        self.assertEqual(best_state, tour)
        self.assertEqual(best_cost, 0)

    def test_performance_small(self) -> None:
        # Simple benchmark to ensure runtime stays reasonable (<0.5s)
        n = 8
        rng = random.Random(123)
        mat = [[0 if i == j else rng.uniform(1, 10) for j in range(n)] for i in range(n)]
        # Symmetrize
        for i in range(n):
            for j in range(i + 1, n):
                w = (mat[i][j] + mat[j][i]) / 2
                mat[i][j] = mat[j][i] = w
        g = Graph(mat)

        sa = SimulatedAnnealing(
            initial_temp=500,
            cooling_rate=0.90,
            iteration_per_temp=150,
            min_temp=1e-3,
            rng=rng,
        )
        start = time.time()
        best_state, best_cost = sa.optimize(
            random_tour(g.n, rng),
            lambda s: tour_cost(g, s),
            tsp_neighbor_swap,
        )
        duration = time.time() - start
        self.assertLess(duration, 0.5)  # sanity check for this environment
        self.assertEqual(len(best_state), n)
        self.assertIsInstance(best_cost, float)

if __name__ == "__main__":
    # Demo run on a random 10‑node graph
    demo_rng = random.Random(2023)
    N = 10
    # Generate symmetric distance matrix with values in [1,20]
    raw = [[0 if i == j else demo_rng.uniform(1, 20) for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            w = (raw[i][j] + raw[j][i]) / 2
            raw[i][j] = raw[j][i] = w
    graph = Graph(raw)

    sa = SimulatedAnnealing(
        initial_temp=1000,
        cooling_rate=0.92,
        iteration_per_temp=300,
        min_temp=1e-4,
        rng=demo_rng,
    )
    init = random_tour(N, demo_rng)
    best_tour, best_len = sa.optimize(
        init,
        lambda s: tour_cost(graph, s),
        tsp_neighbor_swap,
    )
    print("Best tour found:", best_tour)
    print("Tour length :", best_len)
