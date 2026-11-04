import math
import random
from typing import Callable, List, Tuple, Dict, Any, Optional

State = Any
Energy = float
NeighborFunc = Callable[[State, random.Random], State]
EnergyFunc = Callable[[State], Energy]
TemperatureFunc = Callable[[int], float]


class Graph:
    """Undirected weighted graph represented by a distance matrix."""
    def __init__(self, distances: List[List[float]]) -> None:
        n = len(distances)
        if any(len(row) != n for row in distances):
            raise ValueError("Distance matrix must be square.")
        self.n = n
        self._dist = distances

    def distance(self, i: int, j: int) -> float:
        return self._dist[i][j]


def total_distance(tour: List[int], graph: Graph) -> float:
    """Compute total length of a cyclic tour."""
    if len(tour) != graph.n:
        raise ValueError("Tour length must equal number of nodes.")
    dist = 0.0
    for a, b in zip(tour, tour[1:] + [tour[0]]):
        dist += graph.distance(a, b)
    return dist


def two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    """Perform a 2‑opt move: reverse the segment between i and k (inclusive)."""
    if i >= k:
        raise ValueError("i must be less than k for 2‑opt.")
    new_tour = tour[:i] + list(reversed(tour[i:k + 1])) + tour[k + 1:]
    return new_tour


def tsp_neighbor(tour: List[int], rng: random.Random) -> List[int]:
    """Generate a neighboring tour using a random 2‑opt move."""
    n = len(tour)
    i, k = sorted(rng.sample(range(n), 2))
    # Ensure i != k and i != 0 to keep start node stable (optional)
    if i == 0:
        i = 1
    if k == 0:
        k = 1
    if i >= k:
        i, k = k, i
    return two_opt_swap(tour, i, k)


def exponential_cooling(initial_temp: float, cooling_rate: float) -> TemperatureFunc:
    """Return a temperature schedule T(t) = initial_temp * cooling_rate**t."""
    if not (0.0 < cooling_rate < 1.0):
        raise ValueError("cooling_rate must be in (0,1).")
    def schedule(t: int) -> float:
        return initial_temp * (cooling_rate ** t)
    return schedule


class SimulatedAnnealing:
    """Generic Simulated Annealing optimizer."""
    def __init__(
        self,
        initial_state: State,
        neighbor_func: NeighborFunc,
        energy_func: EnergyFunc,
        temperature_func: TemperatureFunc,
        max_steps: int,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.state = initial_state
        self.neighbor = neighbor_func
        self.energy = energy_func
        self.temp = temperature_func
        self.max_steps = max_steps
        self.rng = rng or random.Random()
        self.best_state = initial_state
        self.best_energy = self.energy(initial_state)

    def run(self) -> Tuple[State, Energy]:
        current_state = self.state
        current_energy = self.energy(current_state)
        best_state = current_state
        best_energy = current_energy

        for step in range(self.max_steps):
            T = self.temp(step)
            if T <= 0:
                break
            candidate = self.neighbor(current_state, self.rng)
            candidate_energy = self.energy(candidate)
            delta = candidate_energy - current_energy

            accept = False
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta / T)
                if self.rng.random() < prob:
                    accept = True

            if accept:
                current_state = candidate
                current_energy = candidate_energy
                if current_energy < best_energy:
                    best_state = current_state
                    best_energy = current_energy

        self.best_state = best_state
        self.best_energy = best_energy
        return best_state, best_energy


def random_tour(n: int, rng: random.Random) -> List[int]:
    """Generate a random Hamiltonian tour of n nodes."""
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def euclidean_graph(points: List[Tuple[float, float]]) -> Graph:
    """Create a complete graph where edge weights are Euclidean distances."""
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            d = math.hypot(xi - xj, yi - yj)
            dist[i][j] = dist[j][i] = d
    return Graph(dist)
