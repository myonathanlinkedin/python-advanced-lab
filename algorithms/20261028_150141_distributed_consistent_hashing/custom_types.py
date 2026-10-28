from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    """
    Represents a physical node in the distributed system.
    """
    id: str
