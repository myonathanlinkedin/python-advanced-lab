from __future__ import annotations

import math
from typing import Dict, List

from .types import AllocationError, Block, DoubleFreeError


class BuddyAllocator:
    """
    A simple buddy memory allocator.

    The allocator manages a contiguous memory region of size 2**max_order.
    All allocations are rounded up to the nearest power of two (minimum block size = 2**min_order).
    """

    def __init__(self, max_order: int, min_order: int = 0) -> None:
        """
        Initialise the allocator.

        :param max_order: The exponent of the total memory size (total size = 2**max_order).
        :param min_order: The smallest block exponent allowed (default 0 -> size 1).
        """
        if max_order < min_order:
            raise ValueError("max_order must be >= min_order")
        self.max_order: int = max_order
        self.min_order: int = min_order
        self.total_size: int = 1 << max_order

        # Free lists: order -> list of offsets (stack semantics)
        self._free_lists: Dict[int, List[int]] = {order: [] for order in range(min_order, max_order + 1)}
        # Initially the whole memory is a single free block of max_order
        self._free_lists[max_order].append(0)

        # Allocated blocks: offset -> order
        self._allocated: Dict[int, int] = {}

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def _next_power_of_two(x: int) -> int:
        """Return the smallest power of two >= x (x > 0)."""
        return 1 << (x - 1).bit_length()

    def _order_for_size(self, size: int) -> int:
        """Return the minimal order that can accommodate `size` bytes."""
        if size <= 0:
            raise ValueError("Requested size must be positive")
        needed = max(size, 1 << self.min_order)
        order = (needed - 1).bit_length()
        if order < self.min_order:
            order = self.min_order
        return order

    def _find_suitable_order(self, needed_order: int) -> int:
        """Find the smallest free order >= needed_order; raise if none."""
        for order in range(needed_order, self.max_order + 1):
            if self._free_lists[order]:
                return order
        raise AllocationError(f"No free block large enough for order {needed_order}")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def allocate(self, size: int) -> int:
        """
        Allocate a block of at least `size` bytes.

        :return: The offset (address) of the allocated block.
        :raises AllocationError: If the request cannot be satisfied.
        """
        needed_order = self._order_for_size(size)
        if needed_order > self.max_order:
            raise AllocationError("Requested size exceeds total memory")

        # Find a free block of sufficient order
        order = self._find_suitable_order(needed_order)
        offset = self._free_lists[order].pop()

        # Split until we reach the desired order
        while order > needed_order:
            order -= 1
            buddy_offset = offset + (1 << order)
            self._free_lists[order].append(buddy_offset)  # right buddy becomes free
            # left buddy (offset) continues to be split

        # Record allocation
        self._allocated[offset] = needed_order
        return offset

    def free(self, offset: int) -> None:
        """
        Free a previously allocated block.

        :param offset: The offset returned by a prior allocate().
        :raises DoubleFreeError: If the block was not allocated or already freed.
        """
        if offset not in self._allocated:
            raise DoubleFreeError(f"Attempt to free unallocated or already freed block at offset {offset}")

        order = self._allocated.pop(offset)

        # Coalescing loop
        while order < self.max_order:
            buddy = offset ^ (1 << order)
            buddy_list = self._free_lists[order]
            try:
                buddy_index = buddy_list.index(buddy)
            except ValueError:
                # Buddy not free; cannot merge
                break
            # Remove buddy from free list
            del buddy_list[buddy_index]
            # Determine new merged block's offset (the lower address)
            offset = min(offset, buddy)
            order += 1  # Move to next higher order

        # Insert the (possibly merged) block back into free list
        self._free_lists[order].append(offset)

    # --------------------------------------------------------------------- #
    # Introspection utilities (useful for testing)
    # --------------------------------------------------------------------- #
    def dump_state(self) -> Dict[str, Dict[int, List[int]]]:
        """Return a snapshot of internal state for debugging or testing."""
        return {
            "free_lists": {order: lst.copy() for order, lst in self._free_lists.items()},
            "allocated": self._allocated.copy(),
        }

    def total_free_memory(self) -> int:
        """Calculate total free memory in bytes."""
        return sum((1 << order) * len(lst) for order, lst in self._free_lists.items())

    def total_allocated_memory(self) -> int:
        """Calculate total allocated memory in bytes."""
        return sum(1 << order for order in self._allocated.values())
