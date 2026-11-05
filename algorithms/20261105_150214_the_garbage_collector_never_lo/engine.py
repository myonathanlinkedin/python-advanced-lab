from __future__ import annotations
from typing import Dict, List, Any
from custom_types import RefCounted

# Global registry of live objects. Keys are ``id(obj)`` to avoid accidental
# strong references that would keep objects alive.
_live_registry: Dict[int, RefCounted] = {}


def register(obj: RefCounted) -> None:
    """Add a newly created object to the live registry."""
    _live_registry[id(obj)] = obj


def deregister(obj: RefCounted) -> None:
    """Remove a finalized object from the live registry."""
    _live_registry.pop(id(obj), None)


def inc_ref(obj: RefCounted) -> None:
    """Public API to increase reference count; also ensures registration."""
    if obj.finalized:
        raise RuntimeError("Cannot inc_ref a finalized object")
    obj.inc_ref()
    register(obj)  # Ensure the object is tracked while it has references


def dec_ref(obj: RefCounted) -> None:
    """Public API to decrease reference count; deregisters if finalized."""
    obj.dec_ref()
    if obj.finalized:
        deregister(obj)


def collect_garbage() -> None:
    """
    The collector *never* looks for garbage.
    This function is intentionally a no‑op to satisfy the specification.
    """
    pass  # No tracing, no cycle detection


def get_live_objects() -> List[RefCounted]:
    """Return a snapshot list of currently live (non‑finalized) objects."""
    return list(_live_registry.values())
