import math
import random
from typing import Any, Dict, Iterable, List, Tuple


class CountMinSketch:
    """
    Count‑Min Sketch (CMS) for frequency estimation.

    The sketch maintains a 2‑dimensional array of counters and a list of
    pairwise‑independent hash functions.  It supports:
      * add(item, count=1) – increment the estimated count of *item*.
      * estimate(item) – return the CMS estimate (upper bound) of *item*'s count.
      * merge(other) – in‑place merge another sketch with identical dimensions.
      * heavy_hitters(threshold) – return items whose estimated frequency
        meets or exceeds *threshold*.  The sketch keeps a lightweight
        candidate set of all items ever added; this set is used only for
        heavy‑hitter extraction and does not affect the sketch's error bounds.
    """

    def __init__(
        self,
        width: int,
        depth: int,
        seed: int = 0,
        *,
        store_candidates: bool = True,
    ) -> None:
        """
        Initialise a CMS.

        Parameters
        ----------
        width : int
            Number of columns (the range of each hash function).  Larger width
            reduces the additive error ε·N.
        depth : int
            Number of rows (independent hash functions).  Larger depth reduces
            the failure probability δ.
        seed : int, optional
            Seed for the internal random generator that creates hash seeds.
        store_candidates : bool, optional
            If True, keep a dictionary of all items ever added to enable
            heavy‑hitter queries.  Setting to False saves memory when heavy‑hitter
            detection is not required.
        """
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive integers")
        self.width: int = width
        self.depth: int = depth
        self._table: List[List[int]] = [[0] * width for _ in range(depth)]

        rng = random.Random(seed)
        # Generate a distinct odd integer for each row to act as a hash seed.
        self._hash_seeds: List[int] = [rng.randrange(1, 1 << 31, 2) for _ in range(depth)]

        self._store_candidates: bool = store_candidates
        self._candidates: Dict[Any, int] = {} if store_candidates else {}

    # --------------------------------------------------------------------- #
    # Internal hash function
    # --------------------------------------------------------------------- #
    def _hash(self, item: Any, row: int) -> int:
        """
        Compute a deterministic hash for *item* in the given *row*.
        The built‑in hash() is combined with a per‑row seed and reduced modulo width.
        """
        h = hash(item)
        # Mix the hash with the row‑specific seed using XOR; ensure non‑negative.
        mixed = (h ^ self._hash_seeds[row]) & 0x7FFFFFFFFFFFFFFF
        return mixed % self.width

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def add(self, item: Any, count: int = 1) -> None:
        """
        Increment the count of *item* by *count* (default 1).

        The operation updates one counter per row, chosen by the row's hash
        function.  The time complexity is O(depth).
        """
        if count < 0:
            raise ValueError("count must be non‑negative")
        for row in range(self.depth):
            col = self._hash(item, row)
            self._table[row][col] += count
        if self._store_candidates:
            self._candidates[item] = self._candidates.get(item, 0) + count

    def estimate(self, item: Any) -> int:
        """
        Return the CMS estimate for *item*'s frequency.

        The estimate is the minimum counter among all rows, guaranteeing
        an over‑estimate bounded by ε·N with probability 1‑δ.
        """
        mins = math.inf
        for row in range(self.depth):
            col = self._hash(item, row)
            mins = min(mins, self._table[row][col])
        return mins

    def merge(self, other: "CountMinSketch") -> None:
        """
        Merge *other* into this sketch in‑place.

        Both sketches must share identical dimensions and hash seeds.
        The operation runs in O(width·depth) time.
        """
        if not isinstance(other, CountMinSketch):
            raise TypeError("Can only merge with another CountMinSketch")
        if (self.width, self.depth) != (other.width, other.depth):
            raise ValueError("Sketch dimensions must match for merge")
        if self._hash_seeds != other._hash_seeds:
            raise ValueError("Hash seeds differ; sketches are incompatible")

        for r in range(self.depth):
            row_self = self._table[r]
            row_other = other._table[r]
            for c in range(self.width):
                row_self[c] += row_other[c]

        if self._store_candidates and other._store_candidates:
            for k, v in other._candidates.items():
                self._candidates[k] = self._candidates.get(k, 0) + v

    def heavy_hitters(self, threshold: int) -> List[Tuple[Any, int]]:
        """
        Return a list of (item, estimated_frequency) for all stored items whose
        estimate meets or exceeds *threshold*.

        The method scans the candidate dictionary, which is O(number_of_distinct_items).
        """
        if threshold < 0:
            raise ValueError("threshold must be non‑negative")
        if not self._store_candidates:
            raise RuntimeError("Candidate storage disabled; cannot compute heavy hitters")
        result: List[Tuple[Any, int]] = []
        for item in self._candidates:
            est = self.estimate(item)
            if est >= threshold:
                result.append((item, est))
        return result

    # --------------------------------------------------------------------- #
    # Utility constructors
    # --------------------------------------------------------------------- #
    @classmethod
    def with_error(cls, epsilon: float, delta: float, seed: int = 0) -> "CountMinSketch":
        """
        Factory method that chooses width and depth from error guarantees.

        Parameters
        ----------
        epsilon : float
            Desired additive error (as a fraction of total count N).
        delta : float
            Desired failure probability.
        seed : int, optional
            Random seed for hash function generation.
        """
        if not (0 < epsilon < 1):
            raise ValueError("epsilon must be in (0,1)")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0,1)")
        width = math.ceil(math.e / epsilon)
        depth = math.ceil(math.log(1 / delta))
        return cls(width, depth, seed)
