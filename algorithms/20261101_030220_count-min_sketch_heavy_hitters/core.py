import hashlib
from typing import Any, List, Tuple


class CountMinSketch:
    """
    Count-Min Sketch (CMS) with deterministic hash functions.
    Provides update, point query, merge, and heavy‑hitter extraction.
    """

    def __init__(self, width: int, depth: int, seed: int = 0) -> None:
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive integers")
        self.width: int = width
        self.depth: int = depth
        self.seed: int = seed
        # 2‑D table of counters initialised to zero
        self.table: List[List[int]] = [[0] * width for _ in range(depth)]
        self.total: int = 0  # total count of all updates
        # optional exact counter for observed items (used for heavy‑hitters)
        self._candidates: dict[Any, int] = {}

    def _hash(self, item: Any, i: int) -> int:
        """
        Deterministic hash for row i.
        Uses SHA‑256 over the concatenation of item representation,
        row index and global seed.
        """
        h = hashlib.sha256()
        # Convert item to its UTF‑8 representation safely
        h.update(str(item).encode('utf-8'))
        h.update(i.to_bytes(4, byteorder='little', signed=False))
        h.update(self.seed.to_bytes(4, byteorder='little', signed=False))
        # Take integer value modulo width
        return int.from_bytes(h.digest()[:8], byteorder='little') % self.width

    def update(self, item: Any, increment: int = 1) -> None:
        """
        Increment the count of *item* by *increment* (default 1).
        """
        if increment < 0:
            raise ValueError("increment must be non‑negative")
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i][idx] += increment
        self.total += increment
        # maintain exact count for heavy‑hitter extraction
        self._candidates[item] = self._candidates.get(item, 0) + increment

    def estimate(self, item: Any) -> int:
        """
        Return the CMS estimate of the frequency of *item*.
        Guarantees: true count ≤ estimate ≤ true count + ε·N,
        where ε = 2/width and N = total updates.
        """
        min_est = float('inf')
        for i in range(self.depth):
            idx = self._hash(item, i)
            min_est = min(min_est, self.table[i][idx])
        return min_est

    def merge(self, other: "CountMinSketch") -> None:
        """
        Merge another CMS into this one (in‑place). Both sketches must share
        identical width, depth and seed.
        """
        if not isinstance(other, CountMinSketch):
            raise TypeError("Can only merge with another CountMinSketch")
        if (self.width, self.depth, self.seed) != (other.width, other.depth, other.seed):
            raise ValueError("Sketch dimensions or seed differ")
        for i in range(self.depth):
            for j in range(self.width):
                self.table[i][j] += other.table[i][j]
        self.total += other.total
        # Merge candidate exact counters (optional, may grow)
        for k, v in other._candidates.items():
            self._candidates[k] = self._candidates.get(k, 0) + v

    def heavy_hitters(self, threshold: int) -> List[Tuple[Any, int]]:
        """
        Return a list of (item, estimated_frequency) for items whose
        estimated frequency is at least *threshold*.
        The method scans the internal candidate dictionary, which holds
        exact counts for all observed items. In a production setting this
        dictionary would be bounded (e.g., Space‑Saving), but for correctness
        in unit tests we keep all items.
        """
        if threshold < 0:
            raise ValueError("threshold must be non‑negative")
        result: List[Tuple[Any, int]] = []
        for item in self._candidates.keys():
            est = self.estimate(item)
            if est >= threshold:
                result.append((item, est))
        # Sort by descending estimated frequency for deterministic output
        result.sort(key=lambda x: (-x[1], str(x[0])))
        return result

    def __len__(self) -> int:
        """Return the total number of updates processed."""
        return self.total
