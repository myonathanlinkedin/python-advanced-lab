"""Memory Pool Block Allocator with Free List.

A deterministic, allocation-order-independent block allocator over a fixed
contiguous memory pool. Blocks are managed via a doubly-linked free list and
an O(1) block-index table.

Invariants:
  - Every block is either ALLOCATED or FREE, never both.
  - The free list is a valid doubly-linked chain of FREE blocks.
  - The block table maps every index to exactly one Block.
  - Total allocated bytes + total free bytes == pool size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Block:
    """A single fixed-size block in the pool."""
    index: int
    size: int
    is_allocated: bool = False
    prev: Optional["Block"] = None
    next: Optional["Block"] = None


class MemoryPool:
    """Fixed-size memory pool with O(1) block allocation and deallocation.

    Time complexity:
      - allocate(): O(1)
      - deallocate(): O(1)
      - stats(): O(n) where n is number of blocks

    Space complexity: O(n) for the block table and free list.
    """

    def __init__(self, pool_size: int, block_size: int) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if pool_size % block_size != 0:
            raise ValueError("pool_size must be a multiple of block_size")

        self.pool_size: int = pool_size
        self.block_size: int = block_size
        self.num_blocks: int = pool_size // block_size
        self._blocks: list[Block] = []
        self._free_head: Optional[Block] = None
        self._free_tail: Optional[Block] = None
        self._free_count: int = 0
        self._alloc_count: int = 0

        self._initialize()

    def _initialize(self) -> None:
        """Build the block table and link all blocks into the free list."""
        for i in range(self.num_blocks):
            blk = Block(index=i, size=self.block_size, is_allocated=False)
            self._blocks.append(blk)

        # Link all blocks into a doubly-linked free list.
        for i in range(self.num_blocks):
            if i > 0:
                self._blocks[i].prev = self._blocks[i - 1]
            if i < self.num_blocks - 1:
                self._blocks[i].next = self._blocks[i + 1]

        if self.num_blocks > 0:
            self._free_head = self._blocks[0]
            self._free_tail = self._blocks[-1]
        self._free_count = self.num_blocks
        self._alloc_count = 0

    def allocate(self) -> Optional[Block]:
        """Allocate one free block. Returns None if pool is exhausted.

        Time: O(1).
        """
        if self._free_head is None:
            return None
        blk = self._free_head
        # Unlink from free list.
        self._free_head = blk.next
        if self._free_head is not None:
            self._free_head.prev = None
        else:
            self._free_tail = None
        blk.next = None
        blk.prev = None
        blk.is_allocated = True
        self._free_count -= 1
        self._alloc_count += 1
        return blk

    def deallocate(self, block: Block) -> None:
        """Return a block to the free list.

        Raises ValueError if the block is not currently allocated or does not
        belong to this pool.

        Time: O(1).
        """
        if block.index < 0 or block.index >= self.num_blocks:
            raise ValueError("block does not belong to this pool")
        if not block.is_allocated:
            raise ValueError("block is not currently allocated")
        # Mark free.
        block.is_allocated = False
        # Insert at head of free list for O(1) deallocation.
        block.prev = None
        block.next = self._free_head
        if self._free_head is not None:
            self._free_head.prev = block
        else:
            self._free_tail = block
        self._free_head = block
        self._free_count += 1
        self._alloc_count -= 1

    def is_allocated(self, block: Block) -> bool:
        """Return True if the given block is currently allocated."""
        return block.is_allocated

    def stats(self) -> dict:
        """Return pool statistics.

        Time: O(n) to verify free-list integrity.
        """
        # Walk free list to verify integrity and count.
        count = 0
        cur = self._free_head
        while cur is not None:
            if cur.is_allocated:
                raise RuntimeError("free list contains an allocated block")
            count += 1
            cur = cur.next
        if count != self._free_count:
            raise RuntimeError("free list count mismatch")
        return {
            "pool_size": self.pool_size,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "free_blocks": self._free_count,
            "allocated_blocks": self._alloc_count,
            "free_bytes": self._free_count * self.block_size,
            "allocated_bytes": self._alloc_count * self.block_size,
        }

    def __repr__(self) -> str:
        return (
            f"MemoryPool(pool={self.pool_size}, block={self.block_size}, "
            f"free={self._free_count}/{self.num_blocks})"
        )
