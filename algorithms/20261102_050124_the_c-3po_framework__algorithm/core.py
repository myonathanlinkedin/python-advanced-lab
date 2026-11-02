"""
core.py
---------

Reference implementation of a minimal C‑3PO style framework for handling
#BIS‑hard problems on bipartite graphs with low threshold rank.

The module provides:
* `BipartiteGraph` – a simple bipartite graph representation.
* Exact rank computation of the left‑side adjacency matrix over GF(2).
* Exact counting of independent sets (the #BIS problem) via brute force.
* A placeholder `count_independent_sets_low_rank` that demonstrates the
  intended low‑rank optimisation path while delegating to the exact
  algorithm for correctness.

All code uses only the Python standard library and type hints.
"""

from __future__ import annotations

from typing import List, Tuple


def _popcount(x: int) -> int:
    """Return the number of set bits in a non‑negative integer."""
    return bin(x).count("1")


def _gf2_rank(matrix_rows: List[int], width: int) -> int:
    """
    Compute the rank of a binary matrix over GF(2) using Gaussian elimination.

    Parameters
    ----------
    matrix_rows: List[int]
        Each integer encodes a row as a bit‑vector of length ``width``.
        The most‑significant bit corresponds to column 0.
    width: int
        Number of columns (i.e. the bit‑length of each row).

    Returns
    -------
    int
        Rank of the matrix over GF(2).
    """
    rows = matrix_rows[:]          # work on a copy
    rank = 0
    for col in range(width):
        # mask for the current column (MSB = column 0)
        mask = 1 << (width - 1 - col)
        # Find a pivot row with a 1 in this column
        pivot = None
        for i in range(rank, len(rows)):
            if rows[i] & mask:
                pivot = i
                break
        if pivot is None:
            continue
        # Swap pivot into the current rank position
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_row = rows[rank]
        # Eliminate the current column from all other rows
        for i in range(len(rows)):
            if i != rank and (rows[i] & mask):
                rows[i] ^= pivot_row
        rank += 1
        if rank == len(rows):
            break
    return rank


class BipartiteGraph:
    """
    Simple bipartite graph with left side size ``n_left`` and right side size
    ``n_right``.  Vertices are identified by integers ``0 … n_left-1`` (left)
    and ``0 … n_right-1`` (right).  Edges are stored as bit‑masks for fast
    set operations.
    """

    def __init__(self, n_left: int, n_right: int) -> None:
        if n_left < 0 or n_right < 0:
            raise ValueError("Number of vertices must be non‑negative")
        self.n_left: int = n_left
        self.n_right: int = n_right
        # adjacency from left vertex i to right vertices as an int mask
        self._left_adj: List[int] = [0] * n_left

    def add_edge(self, left: int, right: int) -> None:
        """Add an undirected edge (left, right)."""
        if not (0 <= left < self.n_left):
            raise IndexError("Left vertex out of range")
        if not (0 <= right < self.n_right):
            raise IndexError("Right vertex out of range")
        self._left_adj[left] |= 1 << right

    # --------------------------------------------------------------------- #
    #  Linear‑algebra utilities
    # --------------------------------------------------------------------- #
    def left_adjacency_rows(self) -> List[int]:
        """Return a copy of the left‑side adjacency matrix rows as bit‑vectors."""
        return self._left_adj[:]

    def left_rank_gf2(self) -> int:
        """Rank of the left adjacency matrix over GF(2)."""
        return _gf2_rank(self._left_adj, self.n_right)

    # --------------------------------------------------------------------- #
    #  #BIS – counting independent sets
    # --------------------------------------------------------------------- #
    def count_independent_sets_bruteforce(self) -> int:
        """
        Exact count of independent sets using the classic O(2^{n_left}) algorithm.

        For each subset of left vertices we compute the set of forbidden right
        vertices (their neighbourhood) and count all possible subsets of the
        remaining right vertices.
        """
        total = 0
        full_right_mask = (1 << self.n_right) - 1
        for left_mask in range(1 << self.n_left):
            # Union of neighbours of the selected left vertices
            neigh_mask = 0
            lm = left_mask
            idx = 0
            while lm:
                if lm & 1:
                    neigh_mask |= self._left_adj[idx]
                lm >>= 1
                idx += 1
            # Right vertices that are *not* adjacent to the chosen left set
            free_right = full_right_mask ^ neigh_mask
            total += 1 << _popcount(free_right)
        return total

    def count_independent_sets_low_rank(self, rank_threshold: int = 12) -> int:
        """
        Count independent sets using the low‑rank optimisation path.

        If the GF(2) rank of the left adjacency matrix does not exceed
        ``rank_threshold`` we enumerate subsets of a basis instead of all
        left vertices.  The current implementation falls back to the exact
        brute‑force method because the OR‑based neighbour computation is not
        linear; however the control flow demonstrates the intended design.

        Parameters
        ----------
        rank_threshold: int
            Maximum rank for which the low‑rank routine is considered
            applicable.  The default (12) keeps the exponential factor
            comfortably small for unit tests.

        Returns
        -------
        int
            Exact number of independent sets.
        """
        rank = self.left_rank_gf2()
        if rank > rank_threshold:
            # Rank too high – use the generic exact method.
            return self.count_independent_sets_bruteforce()

        # ---- Low‑rank path (conceptual) ---------------------------------
        # Compute a basis of the adjacency rows.
        basis: List[int] = []
        rows = self._left_adj[:]
        width = self.n_right
        for col in range(width):
            mask = 1 << (width - 1 - col)
            pivot = None
            for i in range(len(rows)):
                if rows[i] & mask:
                    pivot = i
                    break
            if pivot is None:
                continue
            pivot_row = rows[pivot]
            basis.append(pivot_row)
            # Eliminate the pivot column from the remaining rows
            for j in range(len(rows)):
                if j != pivot and (rows[j] & mask):
                    rows[j] ^= pivot_row
            # Remove the used row to keep the list small
            rows.pop(pivot)

        # Enumerate all subsets of the basis (2^rank possibilities).
        # For each subset we compute the neighbour mask and the number of
        # left‑side subsets that realise exactly this mask.
        # The mapping from left subsets to neighbour masks is not linear,
        # therefore we conservatively fall back to the exact count.
        # This placeholder keeps the implementation correct while showing
        # where a sophisticated rank‑based DP would be inserted.
        return self.count_independent_sets_bruteforce()


# End of core.py
# --------------------------------------------------------------------- #
