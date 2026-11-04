import unittest
import random
from typing import List, Tuple

from core import (
    Graph,
    total_distance,
    two_opt_swap,
    tsp_neighbor,
    exponential_cooling,
    SimulatedAnnealing,
    random_tour,
    euclidean_graph,
)

class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self) -> None:
        # Fixed seed for reproducibility
        self.rng = random.Random(42)

    def test_graph_distance(self) -> None:
        g = Graph([[0, 1], [1, 0]])
        self.assertEqual(g.distance(0, 1), 1)
        self.assertEqual(g.distance(1, 0), 1)

    def test_total_distance(self) -> None:
        g = Graph([[0, 2, 3],
                   [2, 0, 4],
                   [3, 4, 0]])
        tour = [0, 1, 2]
        self.assertAlmostEqual(total_distance(tour, g), 2 + 4 + 3)

    def test_two_opt_swap(self) -> None:
        tour = [0, 1, 2, 3, 4]
        new = two_opt_swap(tour, 1, 3)
        self.assertEqual(new, [0, 3, 2, 1, 4])

    def test_tsp_neighbor_is_permutation(self) -> None:
        tour = [0, 1, 2, 3, 4]
        neighbor = tsp_neighbor(tour, self.rng)
        self.assertCountEqual(neighbor, tour)  # same elements
        self.assertNotEqual(neighbor, tour)   # should differ most of the time

    def test_sa_improves_simple_instance(self) -> None:
        # Square with unit edges, optimal tour length = 4
        points = [(0,0), (1,0), (1,1), (0,1)]
        g = euclidean_graph(points)
        init = random_tour(g.n, self.rng)
        sa = SimulatedAnnealing(
            initial_state=init,
            neighbor_func=tsp_neighbor,
            energy_func=lambda tour: total_distance(tour, g),
            temperature_func=exponential_cooling(10.0, 0.99),
            max_steps=5000,
            rng=self.rng,
        )
        best_state, best_energy = sa.run()
        self.assertLessEqual(best_energy, 4.0 + 1e-6)  # optimal or better (numerical)

    def test_sa_deterministic_with_seed(self) -> None:
        points = [(0,0), (2,0), (2,2), (0,2)]
        g = euclidean_graph(points)
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        init1 = random_tour(g.n, rng1)
        init2 = random_tour(g.n, rng2)
        self.assertEqual(init1, init2)

        sa1 = SimulatedAnnealing(
            initial_state=init1,
            neighbor_func=tsp_neighbor,
            energy_func=lambda tour: total_distance(tour, g),
            temperature_func=exponential_cooling(5.0, 0.95),
            max_steps=2000,
            rng=rng1,
        )
        sa2 = SimulatedAnnealing(
            initial_state=init2,
            neighbor_func=tsp_neighbor,
            energy_func=lambda tour: total_distance(tour, g),
            temperature_func=exponential_cooling(5.0, 0.95),
            max_steps=2000,
            rng=rng2,
        )
        best1, energy1 = sa1.run()
        best2, energy2 = sa2.run()
        self.assertEqual(best1, best2)
        self.assertAlmostEqual(energy1, energy2)

if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Demo: solve a random Euclidean TSP with 20 cities
    demo_rng = random.Random(2023)
    num_cities = 20
    points: List[Tuple[float, float]] = [
        (demo_rng.uniform(0, 100), demo_rng.uniform(0, 100))
        for _ in range(num_cities)
    ]
    graph = euclidean_graph(points)
    initial = random_tour(graph.n, demo_rng)
    sa = SimulatedAnnealing(
        initial_state=initial,
        neighbor_func=tsp_neighbor,
        energy_func=lambda tour: total_distance(tour, graph),
        temperature_func=exponential_cooling(20.0, 0.995),
        max_steps=10000,
        rng=demo_rng,
    )
    best_tour, best_len = sa.run()
    print("\nDemo result:")
    print(f"Best tour length: {best_len:.3f}")
    print(f"Tour order: {best_tour}")
