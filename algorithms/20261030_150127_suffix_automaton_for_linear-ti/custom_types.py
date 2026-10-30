from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class State:
    """A node of the suffix automaton."""
    length: int                                 # Max length of strings in this equivalence class
    link: Optional[int] = None                  # Suffix link (index of another state)
    next: Dict[str, int] = field(default_factory=dict)  # Transitions: character -> state index
    # For debugging / optional features
    first_pos: Optional[int] = None             # Position of the first occurrence of the longest string
    is_clone: bool = False                      # True if the state was created as a clone
