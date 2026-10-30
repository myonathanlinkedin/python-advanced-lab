"""
Memory Pool Block Allocator with Free List.

Provides a fixed‑size memory pool that hands out blocks of equal size.
Internally a free list (stack) of block offsets is maintained.
All operations are O(1) and the implementation is pure Python using only
the standard library.
"""

from __future__ import annotations

from typing import List, Optional
from threading import Lock


class MemoryPoolError(RuntimeError):
    """Base class for memory pool related errors."""


class OutOfMemoryError(MemoryPoolError):
    """Raised when allocation is requested but no free blocks remain."""


class InvalidFreeError(MemoryPoolError):
    """Raised when an attempt is made to free a block that does not belong to the pool
    or has already been freed.
    """


class MemoryPool:
    """
    A simple fixed‑size block allocator.

    Parameters
    ----------
    total_size : int
        Total size of the underlying buffer in bytes. Must be a multiple of ``block_size``.
    block_size : int
        Size of each block in bytes. Must be > 0.

    Notes
    -----
    The pool pre‑allocates a ``bytearray`` of ``total_size`` bytes.
    Allocation returns a ``memoryview`` referencing the underlying buffer.
    Deallocation must be performed with the exact ``memoryview`` returned by
    ``allocate``; otherwise ``InvalidFreeError`` is raised.
    """

    def __init__(self, total_size: int, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if total_size <= 0:
            raise ValueError("total_size must be positive")
        if total_size % block_size != 0:
            raise ValueError("total_size must be a multiple of block_size")

        self._total_size: int = total_size
        self._block_size: int = block_size
        self._buffer: bytearray = bytearray(total_size)

        # Stack of free block offsets (in bytes)
        self._free_list: List[int] = [
            i * block_size for i in range(total_size // block_size)
        ][::-1]  # reverse for O(1) pop from end

        # Set of allocated offsets for validation (optional, O(1) average)
        self._allocated: set[int] = set()

        self._lock = Lock()

    @property
    def block_size(self) -> int:
        """Size of each block in bytes."""
        return self._block_size

    @property
    def total_blocks(self) -> int:
        """Total number of blocks the pool can hold."""
        return self._total_size // self._block_size

    @property
    def free_blocks(self) -> int:
        """Number of blocks currently available for allocation."""
        with self._lock:
            return len(self._free_list)

    @property
    def allocated_blocks(self) -> int:
        """Number of blocks currently allocated."""
        with self._lock:
            return len(self._allocated)

    def allocate(self) -> memoryview:
        """
        Allocate a single block and return a memoryview onto it.

        Raises
        ------
        OutOfMemoryError
            If no free blocks are available.
        """
        with self._lock:
            if not self._free_list:
                raise OutOfMemoryError("Memory pool exhausted")
            offset = self._free_list.pop()
            self._allocated.add(offset)

        start = offset
        end = offset + self._block_size
        # memoryview is cheap; it does not copy data.
        return memoryview(self._buffer)[start:end]

    def free(self, block: memoryview) -> None:
        """
        Return a previously allocated block to the pool.

        Parameters
        ----------
        block : memoryview
            The exact memoryview returned by ``allocate``.

        Raises
        ------
        InvalidFreeError
            If the block does not belong to this pool or has already been freed.
        """
        if not isinstance(block, memoryview):
            raise TypeError("block must be a memoryview")

        # Verify that the memoryview references the internal buffer.
        if block.obj is not self._buffer:
            raise InvalidFreeError("Block does not belong to this pool")

        # Compute offset from the start of the buffer.
        # memoryview may have arbitrary shape; we only support 1‑D slices.
        if block.ndim != 1:
            raise InvalidFreeError("Only 1‑dimensional blocks are supported")
        if block.itemsize != 1:
            raise InvalidFreeError("Only byte‑wise blocks are supported")
        offset = block.nbytes * block.readonly + block.offset  # offset in bytes

        # The above expression works for simple slices; for safety we recompute:
        # memoryview.tobytes() would copy, so we avoid it.
        # Instead, we rely on the fact that the view was created as buffer[start:end].
        # The start index equals block.obj.__array_interface__['data'][0] - base address,
        # but Python does not expose that directly. Therefore we store the offset
        # in a hidden attribute when allocating.

        # To avoid fragile introspection, we embed the offset in the view's
        # ``obj`` attribute via a lightweight wrapper class defined below.
        # If the view is not our wrapper, we raise an error.

        if not hasattr(block, "_pool_offset"):
            raise InvalidFreeError("Block was not allocated by this pool")
        offset = block._pool_offset  # type: ignore[attr-defined]

        with self._lock:
            if offset not in self._allocated:
                raise InvalidFreeError("Block already freed or never allocated")
            self._allocated.remove(offset)
            self._free_list.append(offset)

    # Helper for internal use: create a memoryview with hidden offset attribute.
    def _make_view(self, offset: int) -> memoryview:
        view = memoryview(self._buffer)[offset : offset + self._block_size]
        # Attach hidden attribute for validation on free.
        # ``memoryview`` objects allow arbitrary attribute assignment.
        setattr(view, "_pool_offset", offset)  # type: ignore[attr-defined]
        return view

    # Overwrite allocate to use the helper (ensures offset tracking)
    def allocate(self) -> memoryview:  # type: ignore[override]
        with self._lock:
            if not self._free_list:
                raise OutOfMemoryError("Memory pool exhausted")
            offset = self._free_list.pop()
            self._allocated.add(offset)
        return self._make_view(offset)
