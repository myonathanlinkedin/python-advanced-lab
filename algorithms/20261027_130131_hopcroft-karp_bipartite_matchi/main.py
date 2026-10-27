import random
import unittest
from itertools import permutations, combinations
from typing import Dict, Set, Tuple

from core import HopcroftKarp, L, R


def brute_force_maximum_matching(
    left: Set[L], right: Set[R], edges: Set[Tuple[L, R]]
) -> Dict[L, R]:
    """
    Exhaustive search for the maximum matching. Suitable only for tiny graphs.
    Returns one of the optimal matchings (left → right).
    """
    best: Dict[L, R] = {}
    # Try all possible sizes from min(|L|,|R|) down to 0
    max_possible = min(len(left), len(right))
    for size in range(max_possible, -1, -1):
        # Enumerate all subsets of left of given size
        for left_subset in combinations(left, size):
            # For each subset, try all injective mappings to right
            for right_perm in permutations(right, size):
                candidate = dict(zip(left_subset, right_perm))
                if all((u, v) in edges for u, v in candidate.items()):
                    return candidate  # first found of maximal size
    return best


class TestHopcroftKarp(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)

    def _run_random_test(self, n_left: int, n_right: int, p: float) -> None:
        left = {f"L{i}" for i in range(n_left)}
        right = {f"R{j}" for j in range(n_right)}
        hk = HopcroftKarp(left, right)

        edges: Set[Tuple[L, R]] = set()
        for u in left:
            for v in right:
                if random.random() < p:
                    hk.add_edge(u, v)
                    edges.add((u, v))

        matching = hk.maximum_matching()
        # Verify that matching is a valid matching
        self.assertTrue(all((u, v) in edges for u, v in matching.items()))
        self.assertEqual(len(set(matching.values())), len(matching))

        # Compare with brute force for tiny instances
        if n_left <= 5 and n_right <= 5:
            brute = brute_force_maximum_matching(left, right, edges)
            self.assertEqual(len(brute), len(matching))

    def test_empty_graph(self) -> None:
        hk = HopcroftKarp(set(), set())
        self.assertEqual(hk.maximum_matching(), {})

    def test_single_edge(self) -> None:
        hk = HopcroftKarp({1}, {2})
        hk.add_edge(1, 2)
        self.assertEqual(hk.maximum_matching(), {1: 2})

    def test_perfect_matching(self) -> None:
        left = {1, 2, 3}
        right = {'a', 'b', 'c'}
        hk = HopcroftKarp(left, right)
        hk.add_edge(1, 'a')
        hk.add_edge(2, 'b')
        hk.add_edge(3, 'c')
        self.assertEqual(len(hk.maximum_matching()), 3)

    def test_no_matching(self) -> None:
        left = {1, 2}
        right = {'a', 'b'}
        hk = HopcroftKarp(left, right)
        hk.add_edge(1, 'a')
        hk.add_edge(1, 'b')
        # vertex 2 isolated
        self.assertEqual(len(hk.maximum_matching()), 1)

    def test_random_small(self) -> None:
        for _ in range(30):
            self._run_random_test(n_left=4, n_right=4, p=0.4)

    def test_random_medium(self) -> None:
        for _ in range(10):
            self._run_random_test(n_left=8, n_right=10, p=0.3)

    def test_invalid_edge(self) -> None:
        hk = HopcroftKarp({1}, {2})
        with self.assertRaises(ValueError):
            hk.add_edge(1, 3)  # right vertex not in partition

    def test_multiple_edges(self) -> None:
        left = {1, 2}
        right = {'a', 'b'}
        hk = HopcroftKarp(left, right)
        hk.add_edge(1, 'a')
        hk.add_edge(1, 'b')
        hk.add_edge(2, 'a')
        hk.add_edge(2, 'b')
        self.assertEqual(len(hk.maximum_matching()), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
