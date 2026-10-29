from __future__ import annotations
import random
import math
from typing import Callable, List, Tuple, Sequence, Any, Optional

State = Any
CostFunc = Callable[[State], float]
NeighborFunc = Callable[[State, random.Random], State]


class Graph:
    """
    Simple undirected weighted graph represented by an adjacency matrix.
    The matrix must be square and symmetric; diagonal entries are zero.
    """

    def __init__(self, matrix: Sequence[Sequence[float]]) -> None:
        if not matrix:
            raise ValueError("Adjacency matrix cannot be empty.")
        n = len(matrix)
        if any(len(row) != n for row in matrix):
            raise ValueError("Adjacency matrix must be square.")
        for i in range(n):
            if matrix[i][i] != 0:
                raise ValueError("Diagonal entries must be zero.")
            for j in range(i + 1, n):
                if matrix[i][j] != matrix[j][i]:
                    raise ValueError("Matrix must be symmetric.")
                if matrix[i][j] < 0:
                    raise ValueError("Edge weights must be non‑negative.")
        self._matrix: List[List[float]] = [list(row) for row in matrix]
        self.n: int = n

    def distance(self, i: int, j: int) -> float:
        return self._matrix[i][j]

    def __repr__(self) -> str:
        return f"Graph(n={self.n})"


def tour_cost(graph: Graph, tour: Sequence[int]) -> float:
    """Total length of a Hamiltonian tour (returns to start)."""
    if len(tour) != graph.n:
        raise ValueError("Tour length must equal number of vertices.")
    cost = 0.0
    for a, b in zip(tour, list(tour)[1:] + [tour[0]]):
        cost += graph.distance(a, b)
    return cost


def random_tour(n: int, rng: random.Random) -> List[int]:
    """Generate a random permutation of vertices 0..n‑1."""
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def tsp_neighbor_swap(tour: List[int], rng: random.Random) -> List[int]:
    """Return a new tour by swapping two distinct positions."""
    a, b = rng.sample(range(len(tour)), 2)
    new_tour = tour.copy()
    new_tour[a], new_tour[b] = new_tour[b], new_tour[a]
    return new_tour


class SimulatedAnnealing:
    """
    Generic Simulated Annealing optimizer.
    """

    def __init__(
        self,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.95,
        iteration_per_temp: int = 100,
        min_temp: float = 1e-3,
        rng: Optional[random.Random] = None,
    ) -> None:
        if initial_temp <= 0:
            raise ValueError("initial_temp must be positive.")
        if not (0 < cooling_rate < 1):
            raise ValueError("cooling_rate must be in (0,1).")
        if iteration_per_temp <= 0:
            raise ValueError("iteration_per_temp must be positive.")
        if min_temp <= 0:
            raise ValueError("min_temp must be positive.")
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iteration_per_temp = iteration_per_temp
        self.min_temp = min_temp
        self.rng = rng or random.Random()

    def optimize(
        self,
        initial_state: State,
        cost_func: CostFunc,
        neighbor_func: NeighborFunc,
    ) -> Tuple[State, float]:
        """Return (best_state, best_cost) found."""
        current_state = initial_state
        current_cost = cost_func(current_state)
        best_state = current_state
        best_cost = current_cost

        temp = self.initial_temp

        while temp > self.min_temp:
            for _ in range(self.iteration_per_temp):
                candidate = neighbor_func(current_state, self.rng)
                candidate_cost = cost_func(candidate)
                delta = candidate_cost - current_cost

                # Accept if better or with Metropolis probability
                if delta < 0 or self.rng.random() < math.exp(-delta / temp):
                    current_state, current_cost = candidate, candidate_cost
                    if current_cost < best_cost:
                        best_state, best_cost = current_state, current_cost
            temp *= self.cooling_rate

        return best_state, best_cost
