from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Tuple, List, Union, Callable, Any
import struct
import threading


class SharedBuffer:
    """
    A zero‑copy shared memory buffer backed by a mutable bytearray.
    Provides thread‑safe write operations and lock‑free read access via memoryview.
    """

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("size must be positive")
        self._size: int = size
        self._buffer: bytearray = bytearray(size)
        self._lock = threading.RLock()
        # expose a read‑only memoryview for zero‑copy access
        self._view: memoryview = memoryview(self._buffer)

    @property
    def size(self) -> int:
        return self._size

    def get_view(self) -> memoryview:
        """
        Return a zero‑copy memoryview of the underlying buffer.
        The view is read‑only; attempts to modify raise a TypeError.
        """
        return self._view

    def write(self, offset: int, data: bytes) -> None:
        """
        Thread‑safe write of raw bytes into the buffer at the given offset.
        """
        if not (0 <= offset <= self._size):
            raise IndexError("offset out of bounds")
        end = offset + len(data)
        if end > self._size:
            raise IndexError("write exceeds buffer size")
        with self._lock:
            self._buffer[offset:end] = data

    def read(self, offset: int, length: int) -> bytes:
        """
        Thread‑safe read of raw bytes from the buffer.
        """
        if not (0 <= offset <= self._size):
            raise IndexError("offset out of bounds")
        end = offset + length
        if end > self._size:
            raise IndexError("read exceeds buffer size")
        with self._lock:
            return bytes(self._buffer[offset:end])

    def write_int(self, offset: int, value: int) -> None:
        """
        Write a 4‑byte signed integer at the given offset.
        """
        data = struct.pack("<i", value)
        self.write(offset, data)

    def read_int(self, offset: int) -> int:
        """
        Read a 4‑byte signed integer from the given offset.
        """
        raw = self.read(offset, 4)
        return struct.unpack("<i", raw)[0]


class Instruction(NamedTuple):
    """
    Simple bytecode instruction for the Chronos interpreter.
    op: operation name
    args: tuple of integer arguments (offsets or immediate values)
    """
    op: str
    args: Tuple[int, ...]
