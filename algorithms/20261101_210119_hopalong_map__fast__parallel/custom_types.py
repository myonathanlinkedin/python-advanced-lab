from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Generator

@dataclass(frozen=True, slots=True)
class HopalongParams:
    """Parameters (a, b, c) of the Hopalong map."""
    a: float
    b: float
    c: float

@dataclass(frozen=True, slots=True)
class Point:
    """A point (x, y) in the orbit."""
    x: float
    y: float

# Alias for a checkpoint: (step_index, state)
Checkpoint = Tuple[int, Point]

# Alias for a generator yielding points
OrbitGenerator = Generator[Point, None, None]

# Alias for a full orbit stored in memory (used only in tests)
Orbit = List[Point]
