from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


class AllocationError(RuntimeError):
    """Raised when a memory allocation request cannot be satisfied."""
    pass


class DoubleFreeError(RuntimeError):
    """Raised when an attempt is made to free an already freed block."""
    pass


@dataclass(frozen=True)
class Block:
    """Immutable representation of a memory block."""
    offset: int  # Starting address (inclusive)
    order: int   # Order of the block; size = 2**order
