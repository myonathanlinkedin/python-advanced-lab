from __future__ import annotations
import random
import sys
from typing import List

from .engine import sort_by_four_cycles
from .types import OperationLog

def _is_sorted(arr: List[int]) -> bool:
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def _test_random_cases(trials: int = 200) -> None:
    for _ in range(trials):
        size = random.randint(0, 12)
        arr = [random.randint(-100, 100) for _ in range(size)]
        original = list(arr)
        ops: OperationLog = sort_by_four_cycles(arr)
        assert _is_sorted(arr), f"Array not sorted: {original} -> {arr}"
        # Verify that the multiset of elements is unchanged.
        assert sorted(original) == arr, "Element multiset altered during sorting"
        # Verify that every recorded operation is a valid FourCycle.
        n = len(original)
        for op in ops:
            assert len(op) == 4, "Operation does not contain four indices"
            assert len(set(op)) == 4, f"Indices in operation {op} are not distinct"
            for idx in op:
                assert 0 <= idx < n, f"Index {idx} out of bounds for array of size {n}"

def _test_edge_cases() -> None:
    # Empty list
    arr: List[int] = []
    ops = sort_by_four_cycles(arr)
    assert arr == [] and ops == []

    # Single element
    arr = [42]
    ops = sort_by_four_cycles(arr)
    assert arr == [42] and ops == []

    # Two elements (requires fallback to built‑in sort)
    arr = [5, 3]
    ops = sort_by_four_cycles(arr)
    assert arr == [3, 5] and ops == []

    # Three elements (fallback)
    arr = [2, 3, 1]
    ops = sort_by_four_cycles(arr)
    assert arr == [1, 2, 3] and ops == []

    # Already sorted longer list
    arr = list(range(10))
    ops = sort_by_four_cycles(arr)
    assert arr == list(range(10))

if __name__ == "__main__":
    # Run a deterministic demo.
    demo_arr = [9, 1, 4, 7, 3, 6, 2, 5, 8, 0]
    print("Original array:", demo_arr)
    ops = sort_by_four_cycles(demo_arr)
    print("Sorted array :", demo_arr)
    print(f"Number of FourCycle operations applied: {len(ops)}")
    # Run exhaustive internal tests.
    _test_edge_cases()
    _test_random_cases()
    print("All unit tests passed.")
