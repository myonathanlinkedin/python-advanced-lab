from __future__ import annotations
import unittest
import random
from typing import List, Set, Dict
from core import Record, validate, approximate_hitting_set, exact_hitting_set, harmonic


class TestDataValidation(unittest.TestCase):
    def setUp(self):
        # Create a deterministic small dataset
        self.records: List[Record] = [
            {"id": "1", "value": "a"},
            {"id": "2", "value": "b"},
            {"id": "3", "value": "c"},
            {"id": "4", "value": "d"},
            {"id": "5", "value": "e"},
        ]
        # Constraints: each set must have at least one record remaining
        self.constraints: List[Set[int]] = [
            {0, 1},          # either record 0 or 1
            {1, 2, 3},       # at least one of 1,2,3
            {3, 4},          # either 3 or 4
            {0, 2, 4},       # at least one of 0,2,4
        ]

    def test_validate(self):
        # No deletions: should be valid
        self.assertTrue(validate(self.records, self.constraints, set()))
        # Delete record 1: still valid
        self.assertTrue(validate(self.records, self.constraints, {1}))
        # Delete record 0 and 1: violates first constraint
        self.assertFalse(validate(self.records, self.constraints, {0, 1}))

    def test_exact_solution(self):
        exact = exact_hitting_set(self.records, self.constraints)
        self.assertTrue(validate(self.records, self.constraints, exact))
        # For this instance, the optimal deletion set size is 1
        self.assertEqual(len(exact), 1)

    def test_approximation_bound(self):
        exact = exact_hitting_set(self.records, self.constraints)
        approx = approximate_hitting_set(self.records, self.constraints)
        self.assertTrue(validate(self.records, self.constraints, approx))
        # Compute maximum constraint size d
        d = max(len(c) for c in self.constraints)
        bound = harmonic(d) * len(exact)
        self.assertLessEqual(len(approx), bound + 1e-9)  # allow tiny floating error

    def test_random_small(self):
        # Random small instances to test consistency
        for _ in range(20):
            n = random.randint(1, 6)
            m = random.randint(1, 5)
            records = [{"id": str(i)} for i in range(n)]
            constraints = []
            for _ in range(m):
                size = random.randint(1, n)
                constraints.append(set(random.sample(range(n), size)))
            exact = exact_hitting_set(records, constraints)
            approx = approximate_hitting_set(records, constraints)
            self.assertTrue(validate(records, constraints, approx))
            d = max(len(c) for c in constraints) if constraints else 0
            bound = harmonic(d) * len(exact) if d > 0 else 0
            self.assertLessEqual(len(approx), bound + 1e-9)

    def test_edge_cases(self):
        # No constraints: nothing to delete
        self.assertEqual(approximate_hitting_set([], []), set())
        # All constraints empty: nothing to delete
        self.assertEqual(approximate_hitting_set([{}], [set()]), set())
        # Single constraint with single record
        self.assertEqual(approximate_hitting_set([{}], [{0}]), set())


def demo():
    """
    Simple demo: generate random dataset and constraints, run approximation,
    and print results.
    """
    random.seed(42)
    n = 10
    m = 5
    records = [{"id": str(i), "value": chr(65 + i)} for i in range(n)]
    constraints: List[Set[int]] = []
    for _ in range(m):
        size = random.randint(1, n)
        constraints.append(set(random.sample(range(n), size)))

    print("Records:")
    for idx, rec in enumerate(records):
        print(f"  {idx}: {rec}")

    print("\nConstraints:")
    for idx, c in enumerate(constraints):
        print(f"  {idx}: {sorted(c)}")

    deleted = approximate_hitting_set(records, constraints)
    print("\nApproximate deletion set:", sorted(deleted))
    print("Validation result:", validate(records, constraints, deleted))
    exact = exact_hitting_set(records, constraints)
    print("Exact deletion set (size {}):".format(len(exact)), sorted(exact))
    print("Exact validation result:", validate(records, constraints, exact))
    d = max(len(c) for c in constraints) if constraints else 0
    print(f"Approximation bound H({d}) * |OPT| = {harmonic(d) * len(exact):.4f}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n=== Demo ===")
    demo()
