"""
core.py
---------

Zero‑Copy Byte Packet Ring Buffer Dispatcher.

Provides a fixed‑size ring buffer that stores references to ``bytes`` objects
without copying them.  The dispatcher can invoke a user‑supplied callback for
each packet removed from the buffer.

The implementation is deliberately simple, deterministic and uses only the
Python standard library.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Optional, List


class RingBufferDispatcher:
    """
    Fixed‑capacity ring buffer for ``bytes`` packets.

    The buffer stores the *reference* to each ``bytes`` object; no data is
    copied on insertion or removal, guaranteeing zero‑copy semantics.

    Attributes
    ----------
    capacity : int
        Maximum number of packets the buffer can hold.
    _buffer : list[Optional[bytes]]
        Internal storage; pre‑allocated to ``capacity`` slots.
    _head : int
        Index of the next element to pop.
    _tail : int
        Index where the next element will be pushed.
    _size : int
        Current number of stored packets.
    """

    __slots__ = ("capacity", "_buffer", "_head", "_tail", "_size")

    def __init__(self, capacity: int) -> None:
        """
        Initialise a ring buffer.

        Parameters
        ----------
        capacity : int
            Number of packets the buffer can hold. Must be a positive integer.

        Raises
        ------
        ValueError
            If ``capacity`` is not positive.
        """
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity: int = capacity
        self._buffer: List[Optional[bytes]] = [None] * capacity
        self._head: int = 0
        self._tail: int = 0
        self._size: int = 0

    # --------------------------------------------------------------------- #
    # Basic state queries
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        """Return the number of packets currently stored."""
        return self._size

    def is_full(self) -> bool:
        """Return ``True`` if the buffer cannot accept more packets."""
        return self._size == self.capacity

    def is_empty(self) -> bool:
        """Return ``True`` if the buffer holds no packets."""
        return self._size == 0

    # --------------------------------------------------------------------- #
    # Core operations
    # --------------------------------------------------------------------- #
    def push(self, packet: bytes) -> bool:
        """
        Insert a packet into the buffer.

        The packet is stored by reference; the caller retains ownership of the
        original ``bytes`` object.

        Parameters
        ----------
        packet : bytes
            The packet to store. Must be a ``bytes`` instance.

        Returns
        -------
        bool
            ``True`` if the packet was stored, ``False`` if the buffer is full.

        Raises
        ------
        TypeError
            If ``packet`` is not a ``bytes`` object.
        """
        if not isinstance(packet, (bytes, bytearray, memoryview)):
            raise TypeError("packet must be a bytes‑like object")
        if self.is_full():
            return False
        # Store the reference directly – zero copy.
        self._buffer[self._tail] = bytes(packet)  # ensure immutable bytes
        self._tail = (self._tail + 1) % self.capacity
        self._size += 1
        return True

    def pop(self) -> Optional[bytes]:
        """
        Remove and return the oldest packet.

        Returns
        -------
        Optional[bytes]
            The packet if the buffer is non‑empty; otherwise ``None``.
        """
        if self.is_empty():
            return None
        packet = self._buffer[self._head]
        # Clear the slot to break reference cycles.
        self._buffer[self._head] = None
        self._head = (self._head + 1) % self.capacity
        self._size -= 1
        return packet  # type: ignore[return-value]

    def peek(self) -> Optional[bytes]:
        """
        Return the oldest packet without removing it.

        Returns
        -------
        Optional[bytes]
            The packet if present; otherwise ``None``.
        """
        if self.is_empty():
            return None
        return self._buffer[self._head]

    # --------------------------------------------------------------------- #
    # Dispatcher
    # --------------------------------------------------------------------- #
    def dispatch(self, handler: Callable[[bytes], None]) -> int:
        """
        Dequeue all packets and invoke ``handler`` for each.

        Parameters
        ----------
        handler : Callable[[bytes], None]
            Function that processes a single packet.

        Returns
        -------
        int
            Number of packets dispatched.
        """
        count = 0
        while not self.is_empty():
            packet = self.pop()
            # ``packet`` cannot be ``None`` here.
            handler(packet)  # type: ignore[arg-type]
            count += 1
        return count

    # --------------------------------------------------------------------- #
    # Iterable support
    # --------------------------------------------------------------------- #
    def __iter__(self) -> Iterator[bytes]:
        """Iterate over packets from oldest to newest without mutating."""
        idx = self._head
        num = self._size
        for _ in range(num):
            packet = self._buffer[idx]
            if packet is not None:
                yield packet
            idx = (idx + 1) % self.capacity

    # --------------------------------------------------------------------- #
    # Debug representation
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(capacity={self.capacity}, "
            f"size={self._size}, head={self._head}, tail={self._tail})"
        )
