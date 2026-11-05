from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RefCounted:
    """
    Base class for objects managed by a manual reference‑counting system.
    The collector never performs a tracing pass; objects are reclaimed
    only when their explicit reference count reaches zero.
    """
    _refcount: int = field(init=False, default=0, repr=False)
    finalized: bool = field(init=False, default=False, repr=False)

    def inc_ref(self) -> None:
        """Increase the external reference count."""
        self._refcount += 1

    def dec_ref(self) -> None:
        """Decrease the external reference count and finalize if it hits zero."""
        if self._refcount == 0:
            raise RuntimeError("dec_ref called on object with zero reference count")
        self._refcount -= 1
        if self._refcount == 0:
            self._finalize()

    def _finalize(self) -> None:
        """Mark the object as finalized and deregister it from the engine."""
        self.finalized = True
        # The engine will remove the object from its registry; see engine.py


@dataclass
class Node(RefCounted):
    """
    Simple node that can reference another node, forming arbitrary graphs.
    Reference counting of the child is handled automatically.
    """
    child: Optional[Node] = field(default=None, init=False, repr=False)

    def set_child(self, new_child: Optional[Node]) -> None:
        """Replace the current child with ``new_child`` while maintaining reference counts."""
        old_child = self.child
        if old_child is new_child:
            return  # No change

        # Decrease refcount of the old child (if any)
        if old_child is not None:
            old_child.dec_ref()

        # Assign and increase refcount of the new child (if any)
        self.child = new_child
        if new_child is not None:
            new_child.inc_ref()
