from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Node:
    """
    A state (node) of the suffix automaton.

    Attributes
    ----------
    length: int
        The length of the longest string represented by this state.
    link: Optional[int]
        The suffix link (parent) index; None for the root.
    next: Dict[str, int]
        Transitions: character -> state index.
    occ: int
        Number of end‑pos occurrences of the substrings represented by this state.
    """
    length: int = 0
    link: Optional[int] = None
    next: Dict[str, int] = field(default_factory=dict)
    occ: int = 0


__all__ = ["Node"]
