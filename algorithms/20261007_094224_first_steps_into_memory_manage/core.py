from __future__ import annotations
import weakref
from typing import Any, Dict, Tuple, Optional


class MemoryManager:
    """Simple simulated memory manager with manual reference counting."""

    def __init__(self) -> None:
        self._store: Dict[int, Tuple[Any, int]] = {}
        self._next_id: int = 1

    def allocate(self, value: Any) -> MemoryHandle:
        """Allocate a new object and return a handle."""
        obj_id = self._next_id
        self._next_id += 1
        self._store[obj_id] = (value, 1)  # (value, refcount)
        return MemoryHandle(self, obj_id)

    def _inc_ref(self, obj_id: int) -> None:
        if obj_id in self._store:
            value, count = self._store[obj_id]
            self._store[obj_id] = (value, count + 1)

    def _dec_ref(self, obj_id: int) -> None:
        if obj_id not in self._store:
            return
        value, count = self._store[obj_id]
        count -= 1
        if count <= 0:
            del self._store[obj_id]
        else:
            self._store[obj_id] = (value, count)

    def _get(self, obj_id: int) -> Any:
        return self._store[obj_id][0]

    def _set(self, obj_id: int, value: Any) -> None:
        _, count = self._store[obj_id]
        self._store[obj_id] = (value, count)

    def _refcount(self, obj_id: int) -> int:
        return self._store[obj_id][1] if obj_id in self._store else 0

    def __repr__(self) -> str:
        return f"<MemoryManager objects={len(self._store)}>"


class MemoryHandle:
    """Strong reference to a managed object."""

    __slots__ = ("_manager_ref", "_obj_id")

    def __init__(self, manager: MemoryManager, obj_id: int) -> None:
        self._manager_ref = weakref.ref(manager)
        self._obj_id = obj_id

    def get(self) -> Any:
        manager = self._manager()
        if manager is None:
            raise RuntimeError("MemoryManager no longer exists")
        return manager._get(self._obj_id)

    def set(self, value: Any) -> None:
        manager = self._manager()
        if manager is None:
            raise RuntimeError("MemoryManager no longer exists")
        manager._set(self._obj_id, value)

    def retain(self) -> None:
        manager = self._manager()
        if manager:
            manager._inc_ref(self._obj_id)

    def release(self) -> None:
        manager = self._manager()
        if manager:
            manager._dec_ref(self._obj_id)

    def weak(self) -> WeakMemoryHandle:
        return WeakMemoryHandle(self)

    def _manager(self) -> Optional[MemoryManager]:
        return self._manager_ref()

    def __enter__(self) -> MemoryHandle:
        self.retain()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        # Best‑effort cleanup; ignore if manager already gone.
        manager = self._manager()
        if manager:
            manager._dec_ref(self._obj_id)

    def __repr__(self) -> str:
        manager = self._manager()
        rc = manager._refcount(self._obj_id) if manager else "?"
        return f"<MemoryHandle id={self._obj_id} refcount={rc}>"


class WeakMemoryHandle:
    """Weak reference to a MemoryHandle; does not affect refcount."""

    __slots__ = ("_handle_ref",)

    def __init__(self, handle: MemoryHandle) -> None:
        self._handle_ref = weakref.ref(handle)

    def get(self) -> Any:
        handle = self._handle_ref()
        if handle is None:
            raise ReferenceError("Underlying handle has been garbage collected")
        return handle.get()

    def is_alive(self) -> bool:
        return self._handle_ref() is not None

    def __repr__(self) -> str:
        alive = self.is_alive()
        return f"<WeakMemoryHandle alive={alive}>"


__all__ = ["MemoryManager", "MemoryHandle", "WeakMemoryHandle"]
