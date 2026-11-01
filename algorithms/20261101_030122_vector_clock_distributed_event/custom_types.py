from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Mapping, MutableMapping, Optional

ProcessId = str
Clock = Dict[ProcessId, int]


class ComparisonResult(Enum):
    """Result of comparing two vector clocks."""
    LESS = auto()
    EQUAL = auto()
    GREATER = auto()
    CONCURRENT = auto()
