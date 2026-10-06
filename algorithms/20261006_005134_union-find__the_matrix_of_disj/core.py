"""Disjoint Set (Union‑Find) implementation for a 2‑D matrix.

Provides O(α(N)) amortized operations with path compression and union by size.
"""

from __future__ import annotations
from typing import List, Tuple


class DisjointSetMatrix:
    """Union‑Find structure where each element is a cell (row, col) of a matrix."""

    def __init__(self, rows: int, cols: int) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers")
        self._rows: int = rows
        self._cols: int = cols
        size = rows * cols
        self._parent: List[int] = list(range(size))          # parent[i] = i initially
        self._set_size: List[int] = [1] * size               # size of each root set

    def _index(self, row: int, col: int) -> int:
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise IndexError("cell index out of bounds")
        return row * self._cols + col

    def _root(self, idx: int) -> int:
        # Path compression
        while self._parent[idx] != idx:
            self._parent[idx] = self._parent[self._parent[idx]]
            idx = self._parent[idx]
        return idx

    def find(self, row: int, col: int) -> Tuple[int, int]:
        """Return the coordinates of the root cell for (row, col)."""
        root_idx = self._root(self._index(row, col))
        return divmod(root_idx, self._cols)

    def union(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Merge the sets containing (r1,c1) and (r2,c2). Return True if merged."""
        i1 = self._index(r1, c1)
        i2 = self._index(r2, c2)
        root1 = self._root(i1)
        root2 = self._root(i2)
        if root1 == root2:
            return False
        # Union by size: attach smaller tree under larger one
        if self._set_size[root1] < self._set_size[root2]:
            root1, root2 = root2, root1
        self._parent[root2] = root1
        self._set_size[root1] += self._set_size[root2]
        return True

    def connected(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Return True iff (r1,c1) and (r2,c2) belong to the same set."""
        return self._root(self._index(r1, c1)) == self._root(self._index(r2, c2))

    def size(self, row: int, col: int) -> int:
        """Return the number of cells in the set containing (row, col)."""
        root = self._root(self._index(row, col))
        return self._set_size[root]

    def get_set(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Return a list of all cell coordinates belonging to the set of (row, col)."""
        root = self._root(self._index(row, col))
        members: List[Tuple[int, int]] = []
        for r in range(self._rows):
            base = r * self._cols
            for c in range(self._cols):
                if self._root(base + c) == root:
                    members.append((r, c))
        return members

    def sets(self) -> List[List[Tuple[int, int]]]:
        """Return a list of all disjoint sets, each represented as a list of coordinates."""
        root_to_members: dict[int, List[Tuple[int, int]]] = {}
        for r in range(self._rows):
            base = r * self._cols
            for c in range(self._cols):
                idx = base + c
                root = self._root(idx)
                root_to_members.setdefault(root, []).append((r, c))
        return list(root_to_members.values())

    def __repr__(self) -> str:
        return f"DisjointSetMatrix(rows={self._rows}, cols={self._cols})"
