from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PatternInfo:
    """
    Immutable container for a pattern used in the bit‑parallel Levenshtein algorithm.

    Attributes
    ----------
    pattern: str
        The original pattern string.
    length: int
        Length of the pattern (number of characters).
    peq: Dict[str, int]
        Mapping from character to a bit‑mask where the i‑th bit (0‑based from LSB)
        is 1 iff pattern[i] == character.
    mask: int
        Bit‑mask with the lowest ``length`` bits set to 1. Used to truncate
        intermediate results to the pattern width.
    """
    pattern: str
    length: int
    peq: Dict[str, int]
    mask: int
