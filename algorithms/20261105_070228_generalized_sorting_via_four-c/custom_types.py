from __future__ import annotations
from typing import List, Tuple

# A FourCycle is represented as a tuple of four distinct integer indices.
FourCycle = Tuple[int, int, int, int]

# The operation log records the sequence of FourCycles applied during sorting.
OperationLog = List[FourCycle]
