from __future__ import annotations
from typing import List, Set, Dict, Iterable, Tuple
import math
import itertools

Record = Dict[str, str]  # generic record representation


def validate(records: List[Record], constraints: List[Set[int]], deleted: Set[int]) -> bool:
    """
    Return True if every constraint has at least one record remaining.
    :param records: list of records
    :param constraints: list of constraints, each a set of record indices
    :param deleted: set of indices that are considered deleted
    """
    for c in constraints:
        if not any(idx not in deleted for idx in c):
            return False
    return True


def harmonic(n: int) -> float:
    """Compute the nth harmonic number H(n) = 1 + 1/2 + ... + 1/n."""
    return sum(1.0 / i for i in range(1, n + 1))


def approximate_hitting_set(records: List[Record], constraints: List[Set[int]]) -> Set[int]:
    """
    Greedy approximation for the hitting set problem.
    Deletes records until all constraints are satisfied.
    :return: set of indices of records to delete
    """
    n = len(records)
    remaining: Set[int] = set(range(n))
    deleted: Set[int] = set()

    # Precompute which constraints each record participates in
    record_to_constraints: Dict[int, Set[int]] = {i: set() for i in remaining}
    for ci, c in enumerate(constraints):
        for idx in c:
            if idx in remaining:
                record_to_constraints[idx].add(ci)

    while True:
        # Identify violated constraints
        violated = [ci for ci, c in enumerate(constraints) if not any(idx in remaining for idx in c)]
        if not violated:
            break

        # Score each remaining record by how many violated constraints it satisfies
        score: Dict[int, int] = {}
        for ci in violated:
            for idx in constraints[ci]:
                if idx in remaining:
                    score[idx] = score.get(idx, 0) + 1

        if not score:
            # No remaining record can satisfy any violated constraint
            break

        # Choose record with maximum score
        best = max(score, key=lambda k: score[k])
        remaining.remove(best)
        deleted.add(best)

    return deleted


def exact_hitting_set(records: List[Record], constraints: List[Set[int]]) -> Set[int]:
    """
    Brute-force exact solution for small instances.
    Returns a set of indices to delete that satisfies all constraints.
    """
    n = len(records)
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            deleted = set(combo)
            if validate(records, constraints, deleted):
                return deleted
    return set()
