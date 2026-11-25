from __future__ import annotations
from typing import Dict
from .types import PatternInfo


def _build_pattern_info(pattern: str) -> PatternInfo:
    """
    Construct the PatternInfo data structure for a given pattern.

    Parameters
    ----------
    pattern: str
        The pattern string.

    Returns
    -------
    PatternInfo
        Fully populated immutable structure.
    """
    length = len(pattern)
    mask = (1 << length) - 1 if length > 0 else 0
    peq: Dict[str, int] = {}
    for i, ch in enumerate(pattern):
        # Set bit i (LSB = position 0) for character ch
        peq[ch] = peq.get(ch, 0) | (1 << i)
    return PatternInfo(pattern=pattern, length=length, peq=peq, mask=mask)


class BitParallelLevenshtein:
    """
    Bit‑parallel implementation of the Levenshtein (edit) distance
    based on Myers' algorithm. Works for arbitrary pattern lengths
    using Python's unlimited‑precision integers.
    """

    __slots__ = ("_info",)

    def __init__(self, pattern: str) -> None:
        """
        Pre‑process the pattern.

        Parameters
        ----------
        pattern: str
            The pattern against which distances will be computed.
        """
        self._info = _build_pattern_info(pattern)

    def distance(self, text: str) -> int:
        """
        Compute the Levenshtein distance between the stored pattern and ``text``.

        Parameters
        ----------
        text: str
            The query string.

        Returns
        -------
        int
            The edit distance.
        """
        pat = self._info
        m = pat.length

        # Trivial cases
        if m == 0:
            return len(text)
        if len(text) == 0:
            return m

        # Initialise bit‑vectors
        VP = pat.mask          # All 1s within pattern width
        VN = 0
        curr_dist = m

        for ch in text:
            # Eq holds 1s where pattern character equals current text character
            Eq = pat.peq.get(ch, 0)

            # Core Myers operations
            X = Eq | VN
            D0 = (((X & VP) + VP) ^ VP) | X

            # Compute the carry bits
            HP = VN | ~(D0 | VP)
            HN = D0 & VP

            # Update VP and VN for next iteration
            # Shift left by one, insert 0 at LSB for HP/HN handling
            # The mask ensures we never exceed pattern width
            HP_shift = (HP << 1) | 1
            HN_shift = (HN << 1)

            VP = (HN_shift | ~(D0 | HP_shift)) & pat.mask
            VN = (HP_shift & D0) & pat.mask

            # Update the distance according to the top‑most bit (position m‑1)
            if (HP >> (m - 1)) & 1:
                curr_dist += 1
            if (HN >> (m - 1)) & 1:
                curr_dist -= 1

        return curr_dist
