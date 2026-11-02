from __future__ import annotations

from typing import List

from .engine import BuddyAllocator, AllocationError, DoubleFreeError
from .types import Block


def _run_basic_allocation_tests() -> None:
    allocator = BuddyAllocator(max_order=10)  # 1024 bytes total

    # Allocate a block of 100 bytes -> needs order 7 (128)
    a = allocator.allocate(100)
    assert a == 0, "First allocation should start at offset 0"
    state = allocator.dump_state()
    assert state["allocated"][a] == 7
    assert allocator.total_allocated_memory() == 128

    # Allocate another 200 bytes -> needs order 8 (256)
    b = allocator.allocate(200)
    assert b == 128, "Second allocation should start after first block"
    assert allocator.total_allocated_memory() == 128 + 256

    # Allocate a small block of 8 bytes -> order 3 (8)
    c = allocator.allocate(8)
    assert c == 384, "Third allocation should use the next free region"
    assert allocator.total_allocated_memory() == 128 + 256 + 8

    # Free the middle block and ensure it merges correctly after freeing neighbours
    allocator.free(b)
    assert b not in allocator.dump_state()["allocated"]
    # At this point, free list for order 8 should contain offset 128
    assert 128 in allocator.dump_state()["free_lists"][8]

    # Free the first block; should merge with its buddy (offset 128) into order 9 (512)
    allocator.free(a)
    merged_offset = 0
    assert merged_offset in allocator.dump_state()["free_lists"][9], "Blocks should have merged into order 9"

    # Free the small block; should not affect larger merges
    allocator.free(c)
    assert allocator.total_free_memory() == 1024 - allocator.total_allocated_memory()
    # After freeing all, total allocated should be zero
    assert allocator.total_allocated_memory() == 0

    # Allocate the entire memory in one go
    whole = allocator.allocate(1024)
    assert whole == 0
    assert allocator.total_allocated_memory() == 1024
    allocator.free(whole)
    assert allocator.total_allocated_memory() == 0

def _run_edge_case_tests() -> None:
    # Minimal allocator (size 2^4 = 16 bytes)
    allocator = BuddyAllocator(max_order=4, min_order=2)  # min block size = 4 bytes

    # Request exactly the minimum block size
    a = allocator.allocate(4)
    assert a == 0
    assert allocator.total_allocated_memory() == 4

    # Request size that forces rounding up to next power of two (5 -> 8)
    b = allocator.allocate(5)
    assert b == 4
    assert allocator.total_allocated_memory() == 12

    # Exhaust memory
    c = allocator.allocate(4)  # last 4-byte block
    assert c == 12
    try:
        allocator.allocate(1)
        assert False, "Allocation should have failed due to lack of memory"
    except AllocationError:
        pass

    # Double free detection
    allocator.free(a)
    try:
        allocator.free(a)
        assert False, "Double free should raise DoubleFreeError"
    except DoubleFreeError:
        pass

    # Free remaining blocks and ensure full coalescence
    allocator.free(b)
    allocator.free(c)
    assert allocator.total_free_memory() == 16

def _run_randomized_stress_test() -> None:
    import random

    allocator = BuddyAllocator(max_order=12)  # 4096 bytes
    allocated_offsets: List[int] = []
    allocated_sizes: List[int] = []

    for _ in range(1000):
        action = random.choice(["alloc", "free"])
        if action == "alloc" or not allocated_offsets:
            size = random.randint(1, 1024)
            try:
                off = allocator.allocate(size)
                allocated_offsets.append(off)
                allocated_sizes.append(size)
            except AllocationError:
                # Expected when memory is exhausted; ignore
                pass
        else:
            idx = random.randrange(len(allocated_offsets))
            off = allocated_offsets.pop(idx)
            allocator.free(off)
            allocated_sizes.pop(idx)

    # After random operations, free everything left
    for off in allocated_offsets[:]:
        allocator.free(off)
    assert allocator.total_allocated_memory() == 0
    assert allocator.total_free_memory() == allocator.total_size

def main() -> None:
    _run_basic_allocation_tests()
    _run_edge_case_tests()
    _run_randomized_stress_test()
    print("All BuddyAllocator tests passed successfully.")

if __name__ == "__main__":
    main()
