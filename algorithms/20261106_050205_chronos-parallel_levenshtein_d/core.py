"""
core.py
---------

Implementation of a stratum‑matrix (anti‑diagonal) parallel Levenshtein distance
algorithm.  The algorithm computes the classic edit distance but processes
each anti‑diagonal (stratum) independently, allowing parallel evaluation of
the cells within the stratum.

The implementation uses only the Python standard library and is deterministic:
all reads for a cell are from previously completed strata, guaranteeing
thread‑safety without locks.

Public API
----------
- levenshtein_parallel(s: str, t: str) -> int
    Compute the Levenshtein distance between *s* and *t* using the
    stratum‑matrix parallel approach.

- levenshtein_naive(s: str, t: str) -> int
    Classic O(m·n) DP implementation used for correctness verification.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


def _cell_compute(i: int, j: int, s: str, t: str, dp: List[List[int]]) -> Tuple[int, int, int]:
    """
    Compute the DP value for cell (i, j) where i > 0 and j > 0.

    Returns a tuple (i, j, value) so the caller can assign it back into the matrix.
    """
    cost = 0 if s[i - 1] == t[j - 1] else 1
    deletion = dp[i - 1][j] + 1          # from (i-1, j)
    insertion = dp[i][j - 1] + 1         # from (i, j-1)
    substitution = dp[i - 1][j - 1] + cost  # from (i-1, j-1)
    value = min(deletion, insertion, substitution)
    return i, j, value


def levenshtein_parallel(s: str, t: str) -> int:
    """
    Compute the Levenshtein distance between strings *s* and *t* using a
    stratum‑matrix parallel algorithm.

    The algorithm proceeds over anti‑diagonals (i + j = const).  All cells
    belonging to the same anti‑diagonal are independent and are evaluated
    concurrently using a ThreadPoolExecutor.

    Time   : O(|s|·|t|)   (same as classic DP)
    Space  : O(|s|·|t|)   (full matrix required for random access)
    Parallelism : up to min(|s|, |t|) workers per stratum.
    """
    m, n = len(s), len(t)

    # Edge cases: one of the strings is empty.
    if m == 0:
        return n
    if n == 0:
        return m

    # Initialise DP matrix (m+1) x (n+1)
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    # Base rows/columns
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    # Process each stratum (anti‑diagonal) where i + j = k, k ranges from 2 .. m+n
    max_k = m + n
    # Choose a reasonable default for max workers; using min(m, n) caps it.
    max_workers = min(m, n, 32)  # 32 is an arbitrary upper bound to avoid oversubscription

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for k in range(2, max_k + 1):
            # Determine valid (i, j) pairs for this stratum.
            # i ranges from max(1, k-n) to min(m, k-1)
            i_start = max(1, k - n)
            i_end = min(m, k - 1)
            tasks: List[Tuple[int, int]] = [(i, k - i) for i in range(i_start, i_end + 1)]

            # Submit all cell computations for this stratum.
            futures = [executor.submit(_cell_compute, i, j, s, t, dp) for i, j in tasks]

            # Collect results and write them back into the matrix.
            for future in futures:
                i, j, value = future.result()
                dp[i][j] = value

    return dp[m][n]


def levenshtein_naive(s: str, t: str) -> int:
    """
    Classic O(m·n) dynamic‑programming Levenshtein distance implementation.
    Used for correctness testing.
    """
    m, n = len(s), len(t)
    if m == 0:
        return n
    if n == 0:
        return m

    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        si = s[i - 1]
        for j in range(1, n + 1):
            cost = 0 if si == t[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # deletion
                dp[i][j - 1] + 1,          # insertion
                dp[i - 1][j - 1] + cost    # substitution
            )
    return dp[m][n]


__all__ = ["levenshtein_parallel", "levenshtein_naive"]
