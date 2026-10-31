from __future__ import annotations

from typing import Dict, List, Tuple


class MemoryPoolError(Exception):
    """Base exception for MemoryPool errors."""


class InvalidHandleError(MemoryPoolError):
    """Raised when an operation references an unknown handle."""


class DoubleFreeError(MemoryPoolError):
    """Raised when a handle is freed more than once."""


class MemoryPool:
    """
    A simple memory pool that reuses fixed‑size blocks.
    It tracks the amount of memory obtained from the system and the
    memory saved by reusing previously freed blocks.
    """

    def __init__(self) -> None:
        # Mapping from size -> list of free handles (simulated)
        self._free_lists: Dict[int, List[int]] = {}
        # Mapping from handle -> size for allocated blocks
        self._allocated: Dict[int, int] = {}
        # Counter to generate unique handles
        self._next_handle: int = 1
        # Total bytes ever requested from the system (including reused)
        self._system_memory: int = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def allocate(self, size: int) -> int:
        """
        Allocate a block of the given size.
        Returns a unique handle representing the block.
        """
        if size < 0:
            raise ValueError("size must be non‑negative")
        # Reuse if possible
        free_list = self._free_lists.get(size)
        if free_list:
            handle = free_list.pop()
        else:
            handle = self._next_handle
            self._next_handle += 1
            self._system_memory += size
        self._allocated[handle] = size
        return handle

    def free(self, handle: int) -> None:
        """
        Release a previously allocated block.
        The block becomes available for future allocations of the same size.
        """
        if handle not in self._allocated:
            raise InvalidHandleError(f"Handle {handle} is not allocated")
        size = self._allocated.pop(handle)
        self._free_lists.setdefault(size, []).append(handle)

    def current_memory_in_use(self) -> int:
        """Return the total size of currently allocated (not freed) blocks."""
        return sum(self._allocated.values())

    def system_memory(self) -> int:
        """Return the total memory ever obtained from the system."""
        return self._system_memory

    def memory_saved(self) -> int:
        """
        Return the amount of memory saved by reusing blocks.
        This is the difference between system memory and the sum of
        currently allocated blocks.
        """
        return self._system_memory - self.current_memory_in_use()

    def stats(self) -> Tuple[int, int, int]:
        """
        Return a tuple (system_memory, current_in_use, saved_memory).
        """
        return (
            self.system_memory(),
            self.current_memory_in_use(),
            self.memory_saved(),
        )

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def reset(self) -> None:
        """Reset the pool to its initial empty state."""
        self._free_lists.clear()
        self._allocated.clear()
        self._next_handle = 1
        self._system_memory = 0


def measure_memory_savings(alloc_sequence: List[int]) -> Tuple[int, int, int]:
    """
    Simulate a series of allocations and immediate frees using a MemoryPool.
    The sequence consists of positive integers (allocation sizes) and
    negative integers (free the most recent allocation of that size).

    Returns a tuple (system_memory, final_in_use, saved_memory) where:
      - system_memory is the total bytes requested from the system.
      - final_in_use is the sum of sizes still allocated at the end.
      - saved_memory is the difference between system_memory and final_in_use.
    """
    pool = MemoryPool()
    # Stack to remember allocation order for matching frees
    allocation_stack: List[Tuple[int, int]] = []  # (handle, size)

    for entry in alloc_sequence:
        if entry >= 0:
            handle = pool.allocate(entry)
            allocation_stack.append((handle, entry))
        else:
            # Negative entry means free the most recent allocation of size -entry
            target_size = -entry
            # Find the most recent matching allocation
            for i in range(len(allocation_stack) - 1, -1, -1):
                h, sz = allocation_stack[i]
                if sz == target_size:
                    pool.free(h)
                    allocation_stack.pop(i)
                    break
            else:
                raise InvalidHandleError(
                    f"No allocated block of size {target_size} to free"
                )

    return pool.stats()
