"""
Bit‑parallel Levenshtein distance (Myers' algorithm).

Provides:
    levenshtein_bitparallel(s: str, t: str) -> int
"""

from __future__ import annotations
from typing import Dict


def _preprocess_pattern(pat: str) -> Dict[str, int]:
    """
    Build a dictionary mapping each character to a bitmask indicating the
    positions (0‑based) where it occurs in *pat*.
    """
    masks: Dict[str, int] = {}
    for i, ch in enumerate(pat):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    return masks


def levenshtein_bitparallel(s: str, t: str) -> int:
    """
    Compute the Levenshtein distance between *s* and *t* using Myers' bit‑parallel
    algorithm. The implementation works for arbitrary length strings because Python
    integers have unlimited precision; the algorithm treats the pattern as a bit‑vector.

    The function always treats the shorter string as the pattern to keep the bit‑vector
    size minimal.
    """
    # Ensure pattern is the shorter string
    if len(s) > len(t):
        s, t = t, s

    m = len(s)
    n = len(t)

    # Trivial cases
    if m == 0:
        return n
    if n == 0:
        return m

    # Pre‑process pattern
    char_mask = _preprocess_pattern(s)

    # Initialise bit‑vectors
    # All 1s for VP (positive), 0 for VN (negative)
    VP = (1 << m) - 1  # mask of m bits set to 1
    VN = 0
    score = m  # initial distance = length of pattern

    mask = (1 << m) - 1  # keep only m bits

    for ch in t:
        # Eq contains 1s where pattern character equals current text character
        Eq = char_mask.get(ch, 0)

        # Step 1: compute D0
        X = Eq | VN
        D0 = (((X & VP) + VP) ^ VP) | X

        # Step 2: compute HP and HN
        HP = VN | ~(D0 | VP)
        HN = D0 & VP

        # Step 3: shift HP and HN left by one, add 1 to HP (as per algorithm)
        HP = ((HP << 1) | 1) & ((1 << (m + 1)) - 1)  # keep m+1 bits
        HN = (HN << 1) & ((1 << (m + 1)) - 1)

        # Step 4: update VP and VN
        VP = (HN | ~(D0 | HP)) & mask
        VN = D0 & HP

        # Step 5: adjust score based on overflow bits
        if (HP >> m) & 1:
            score += 1
        if (HN >> m) & 1:
            score -= 1

    return score
