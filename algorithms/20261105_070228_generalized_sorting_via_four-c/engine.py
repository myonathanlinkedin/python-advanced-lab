from __future__ import annotations
from typing import List, Tuple, Set
from .types import FourCycle, OperationLog

def apply_four_cycle(arr: List[int], i: int, j: int, k: int, l: int) -> None:
    """
    Perform an in‑place left rotation of the four elements at the given indices:
    (arr[i], arr[j], arr[k], arr[l]) ← (arr[l], arr[i], arr[j], arr[k])
    The indices must be distinct; the caller is responsible for this guarantee.
    """
    arr[i], arr[j], arr[k], arr[l] = arr[l], arr[i], arr[j], arr[k]

def _build_target_positions(arr: List[int]) -> dict[int, List[int]]:
    """
    For each distinct value in ``arr`` return a list of target indices where
    that value should appear in the sorted order (stable, left‑to‑right).
    """
    sorted_vals = sorted(arr)
    pos_map: dict[int, List[int]] = {}
    for idx, val in enumerate(sorted_vals):
        pos_map.setdefault(val, []).append(idx)
    return pos_map

def _perm_from_array(arr: List[int]) -> List[int]:
    """
    Compute the permutation ``p`` such that moving the element currently at
    index ``i`` to index ``p[i]`` yields a sorted array.
    """
    target_pos = _build_target_positions(arr)
    # For duplicate values we consume target positions from left to right.
    used: dict[int, int] = {}
    perm = [0] * len(arr)
    for i, val in enumerate(arr):
        used[val] = used.get(val, 0)
        perm[i] = target_pos[val][used[val]]
        used[val] += 1
    return perm

def _decompose_cycle_to_four_cycles(
    cycle: List[int],
    helper: int,
    helpers: Tuple[int, int],
    ops: OperationLog,
    arr: List[int],
) -> None:
    """
    Decompose a permutation cycle into a sequence of FourCycles.
    ``helper`` is a distinguished index that does **not** belong to ``cycle``.
    ``helpers`` supplies a second helper index used for 2‑cycles.
    The generated FourCycles are applied immediately to ``arr`` and recorded
    in ``ops``.
    """
    if len(cycle) == 2:
        a, b = cycle
        h1, h2 = helpers
        # Two 4‑cycles realise a transposition (a b) while leaving h1 and h2
        # unchanged overall.
        apply_four_cycle(arr, a, h1, b, h2)
        ops.append((a, h1, b, h2))
        apply_four_cycle(arr, a, h2, b, h1)
        ops.append((a, h2, b, h1))
    else:
        # For a cycle (c0 c1 … cL‑1) with L ≥ 3 we repeatedly apply
        # (c0, ci, helper, ci‑1) for i = L‑1 … 1.
        c0 = cycle[0]
        for i in range(len(cycle) - 1, 0, -1):
            ci = cycle[i]
            ci_1 = cycle[i - 1]
            apply_four_cycle(arr, c0, ci, helper, ci_1)
            ops.append((c0, ci, helper, ci_1))

def sort_by_four_cycles(arr: List[int]) -> OperationLog:
    """
    Sort ``arr`` **exclusively** by applying FourCycle operations.
    The function returns the ordered list of FourCycles that were applied.
    For arrays shorter than four elements the built‑in ``list.sort`` is used
    (no FourCycles are required).
    """
    n = len(arr)
    ops: OperationLog = []

    if n < 4:
        arr.sort()
        return ops

    # Choose two distinct helper indices that are not part of any
    # non‑trivial cycle whenever possible.
    helper_main = n - 1
    helper_aux = n - 2

    # Compute the permutation that would sort the array.
    perm = _perm_from_array(arr)

    # Decompose the permutation into disjoint cycles.
    visited: Set[int] = set()
    for start in range(n):
        if start in visited or perm[start] == start:
            continue

        # Extract the current cycle.
        cycle: List[int] = []
        i = start
        while i not in visited:
            visited.add(i)
            cycle.append(i)
            i = perm[i]

        # Ensure the chosen helpers are not inside the current cycle.
        # If they are, pick alternative helpers (there are at least two
        # indices outside the cycle because n ≥ 4).
        h_main = helper_main
        h_aux = helper_aux
        if h_main in cycle or h_aux in cycle:
            # Find two indices outside the cycle.
            outsiders = [idx for idx in range(n) if idx not in cycle]
            h_main, h_aux = outsiders[0], outsiders[1]

        _decompose_cycle_to_four_cycles(cycle, h_main, (h_main, h_aux), ops, arr)

    return ops
