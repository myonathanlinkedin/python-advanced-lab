"""
Bit-Parallel Levenshtein Distance Engine
----------------------------------------

This module implements Myers' bit-parallel algorithm for computing the
Levenshtein (edit) distance between two strings.  The algorithm runs in
O(n) time where n is the length of the source string, and uses only
bitwise operations on Python integers, which are of arbitrary size.
"""

from __future__ import annotations
from typing import Dict

__all__ = ["bit_parallel_levenshtein"]


def bit_parallel_levenshtein(s: str, t: str) -> int:
    """
    Compute the Levenshtein distance between strings `s` and `t`
    using Myers' bit-parallel algorithm.

    Parameters
    ----------
    s : str
        Source string (text).
    t : str
        Pattern string (reference).

    Returns
    -------
    int
        The edit distance between `s` and `t`.

    Notes
    -----
    The algorithm assumes that the pattern length `m` fits into a
    Python integer bit-width.  Python integers are unbounded, so the
    implementation works for arbitrarily long patterns, albeit with
    a linear-time cost in the pattern length.

    The implementation follows the pseudocode from Myers (2000) and
    masks intermediate results to `m` bits to avoid sign-extension
    effects from the bitwise NOT operator.
    """
    m = len(t)
    if m == 0:
        return len(s)
    if len(s) == 0:
        return m

    # Preprocess: build a dictionary mapping each character in `t`
    # to a bitmask indicating its positions.
    Peq: Dict[str, int] = {}
    for i, ch in enumerate(t):
        Peq[ch] = Peq.get(ch, 0) | (1 << i)

    # Initialize bit vectors.
    VP = ~0  # All ones
    VN = 0
    score = m
    mask = (1 << m) - 1  # Mask to keep only m bits

    for ch in s:
        Peq_i = Peq.get(ch, 0)
        X = Peq_i | VN
        D0 = (((X & VP) + VP) ^ VP) | X

        HP = VN | ~(D0 | VP)
        HN = VP & D0

        # Mask HP and HN to m bits before using them.
        HP &= mask
        HN &= mask

        # Update score based on the most significant bit.
        if HP & (1 << (m - 1)):
            score += 1
        if HN & (1 << (m - 1)):
            score -= 1

        # Shift operations for next iteration.
        HP_shift = (HP << 1) | 1
        VP = (HN << 1) | ~(D0 | HP_shift)
        VN = HP_shift & D0

        # Keep only m bits.
        VP &= mask
        VN &= mask

    return score
