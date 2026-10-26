from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Tuple, Union

Packet = Union[bytes, bytearray, memoryview]


class RingBuffer:
    """
    Fixed‑size ring buffer that stores *references* to byte‑like packets
    using ``memoryview`` to avoid any data copying.

    The buffer holds at most ``capacity`` packets.  All operations are
    O(1) time and O(1) additional space.
    """

    __slots__ = ("_capacity", "_buf", "_head", "_tail", "_count", "_lock")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity: int = capacity
        self._buf: List[Optional[memoryview]] = [None] * capacity
        self._head: int = 0  # index of next element to pop
        self._tail: int = 0  # index of next slot to push
        self._count: int = 0
        self._lock = threading.Lock()

    # --------------------------------------------------------------------- #
    # Basic state queries
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        with self._lock:
            return self._count

    def is_full(self) -> bool:
        with self._lock:
            return self._count == self._capacity

    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    # --------------------------------------------------------------------- #
    # Core operations
    # --------------------------------------------------------------------- #
    def push(self, packet: Packet) -> bool:
        """
        Store *packet* in the buffer without copying.
        Returns ``True`` on success, ``False`` if the buffer is full.
        """
        mv = memoryview(packet)
        with self._lock:
            if self._count == self._capacity:
                return False
            self._buf[self._tail] = mv
            self._tail = (self._tail + 1) % self._capacity
            self._count += 1
            return True

    def pop(self) -> Optional[memoryview]:
        """
        Retrieve the oldest packet.  Returns ``None`` if the buffer is empty.
        The returned ``memoryview`` shares the original underlying buffer.
        """
        with self._lock:
            if self._count == 0:
                return None
            mv = self._buf[self._head]
            # Help GC – not strictly required but keeps references tidy.
            self._buf[self._head] = None
            self._head = (self._head + 1) % self._capacity
            self._count -= 1
            return mv

    # --------------------------------------------------------------------- #
    # Utility for debugging / introspection
    # --------------------------------------------------------------------- #
    def snapshot(self) -> List[memoryview]:
        """Return a list copy of the current contents (oldest → newest)."""
        with self._lock:
            items: List[memoryview] = []
            idx = self._head
            for _ in range(self._count):
                mv = self._buf[idx]
                assert mv is not None  # invariant
                items.append(mv)
                idx = (idx + 1) % self._capacity
            return items


Callback = Callable[[memoryview], None]


class Dispatcher:
    """
    Dispatches packets stored in a ``RingBuffer`` to registered callbacks.
    The first byte of each packet is interpreted as a *type identifier*.
    """

    __slots__ = ("_ring", "_handlers", "_lock")

    def __init__(self, ring: RingBuffer) -> None:
        self._ring: RingBuffer = ring
        self._handlers: Dict[int, Callback] = {}
        self._lock = threading.Lock()

    # --------------------------------------------------------------------- #
    # Registration API
    # --------------------------------------------------------------------- #
    def register(self, type_id: int, handler: Callback) -> None:
        """
        Associate ``handler`` with ``type_id``.  ``type_id`` must be in 0‑255.
        Overwrites any existing handler for the same ``type_id``.
        """
        if not (0 <= type_id <= 255):
            raise ValueError("type_id must be in range 0..255")
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            self._handlers[type_id] = handler

    def unregister(self, type_id: int) -> None:
        """Remove the handler for ``type_id`` if present."""
        with self._lock:
            self._handlers.pop(type_id, None)

    # --------------------------------------------------------------------- #
    # Dispatching API
    # --------------------------------------------------------------------- #
    def dispatch_one(self) -> bool:
        """
        Pop a single packet from the ring and invoke the matching handler.
        Returns ``True`` if a packet was processed, ``False`` if the ring is empty.
        Raises ``KeyError`` if no handler is registered for the packet's type.
        """
        packet = self._ring.pop()
        if packet is None:
            return False

        if len(packet) == 0:
            raise ValueError("packet is empty; cannot extract type identifier")

        type_id = packet[0]
        with self._lock:
            handler = self._handlers.get(type_id)

        if handler is None:
            raise KeyError(f"No handler registered for type_id {type_id}")

        # The handler receives the *whole* packet (including the type byte).
        handler(packet)
        return True

    def dispatch_all(self) -> int:
        """
        Continuously dispatch packets until the ring buffer is empty.
        Returns the number of packets processed.
        """
        processed = 0
        while self.dispatch_one():
            processed += 1
        return processed
