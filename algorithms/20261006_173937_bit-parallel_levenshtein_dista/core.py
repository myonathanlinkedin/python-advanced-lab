import typing

__all__ = ["bit_levenshtein"]


def _build_bitmask(pattern: str) -> dict[str, int]:
    """Build a bitmask dictionary for the pattern.

    Each character maps to a bitmask where the i-th bit is set if the
    character occurs at position i in the pattern.
    """
    bitmask: dict[str, int] = {}
    for i, ch in enumerate(pattern):
        bitmask.setdefault(ch, 0)
        bitmask[ch] |= 1 << i
    return bitmask


def bit_levenshtein(s: str, t: str, max_distance: typing.Optional[int] = None) -> int:
    """Compute the Levenshtein distance between strings s and t using
    Myers' bit‑parallel algorithm.

    Parameters
    ----------
    s : str
        The source string.
    t : str
        The target string (pattern). Must be at most 64 characters long.
    max_distance : int, optional
        If provided, the function will stop early and return a value
        greater than ``max_distance`` as soon as the distance exceeds it.

    Returns
    -------
    int
        The edit distance between ``s`` and ``t``.
    """
    m = len(t)
    if m == 0:
        return len(s)
    if len(s) == 0:
        return m
    if m > 64:
        raise ValueError("Bit‑parallel algorithm supports patterns up to 64 characters.")

    bitmask = _build_bitmask(t)

    VP = ~0
    VN = 0
    score = m

    for ch in s:
        PM = bitmask.get(ch, 0)
        X = PM | VN
        D0 = (((X & VP) + VP) ^ VP) | X
        HP = VN | ~(D0 | VP)
        HN = VP & D0

        HP = (HP << 1) | 1
        HN = (HN << 1)

        VP = (HN | ~(D0 | HP))
        VN = HP & D0

        if D0 & (1 << (m - 1)):
            score += 1
        elif HP & (1 << (m - 1)):
            score -= 1

        if max_distance is not None and score > max_distance:
            return score

    return score
