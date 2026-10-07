from __future__ import annotations
from typing import Dict

__all__ = ["bit_parallel_levenshtein"]


def _build_char_mask(pattern: str) -> Dict[str, int]:
    """Build a bitmask for each character in the pattern."""
    mask: Dict[str, int] = {}
    for i, ch in enumerate(pattern):
        mask.setdefault(ch, 0)
        mask[ch] |= 1 << i
    return mask


def bit_parallel_levenshtein(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein distance between s1 and s2 using Myers' bit‑parallel algorithm.
    Works for patterns up to 64 characters (size of a Python int bit width).
    Raises ValueError if the pattern length exceeds 64.
    """
    m = len(s2)
    if m == 0:
        return len(s1)
    if len(s1) == 0:
        return m
    if m > 64:
        raise ValueError("Pattern length must be <= 64 for bit‑parallel algorithm.")

    char_mask = _build_char_mask(s2)
    mask = (1 << m) - 1
    VP = mask
    VN = 0
    score = m

    for ch in s1:
        PM = char_mask.get(ch, 0)
        X = PM | VN
        D0 = (((X + VP) ^ VP) | X) & mask
        HP = VN | ~(D0 | VP)
        HN = VP & D0
        if HP & (1 << (m - 1)):
            score += 1
        elif HN & (1 << (m - 1)):
            score -= 1
        VP = ((HN << 1) | ~(D0 | ((HP << 1) | 1))) & mask
        VN = ((HP << 1) & D0) & mask

    return score
