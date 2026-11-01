"""
types.py
Domain models, interfaces, and memory structs for the Buddy Memory Allocation System.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Set


class AllocationStatus(Enum):
    """Status of a memory allocation request."""
    SUCCESS = auto()
    FAILURE = auto()


class BlockState(Enum):
    """State of a memory block in the buddy system."""
    FREE = auto()
    ALLOCATED = auto()


@dataclass
class MemoryBlock:
    """
    Represents a single memory block in the buddy system.
    
    Attributes:
        address: Starting address of the block.
        size: Size of the block (must be a power of 2).
        state: Current state (FREE or ALLOCATED).
        parent: Reference to parent block (for coalescing).
        left_buddy: Reference to left buddy block.
        right_buddy: Reference to right buddy block.
        allocation_id: Unique identifier if allocated, None if free.
    """
    address: int
    size: int
    state: BlockState = BlockState.FREE
    parent: Optional['MemoryBlock'] = None
    left_buddy: Optional['MemoryBlock'] = None
    right_buddy: Optional['MemoryBlock'] = None
    allocation_id: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate block invariants."""
        if self.size <= 0:
            raise ValueError(f"Block size must be positive, got {self.size}")
        if self.size & (self.size - 1) != 0:
            raise ValueError(f"Block size must be a power of 2, got {self.size}")
        if self.address < 0:
            raise ValueError(f"Block address must be non-negative, got {self.address}")

    @property
    def end_address(self) -> int:
        """Return the end address (exclusive) of this block."""
        return self.address + self.size

    @property
    def buddy_address(self) -> int:
        """
        Calculate the address of the buddy block.
        
        For a block at address A with size S:
        - If A is divisible by 2*S, the buddy is at A + S (right buddy)
        - Otherwise, the buddy is at A - S (left buddy)
        """
        if self.address % (2 * self.size) == 0:
            return self.address + self.size
        else:
            return self.address - self.size

    def __repr__(self) -> str:
        state_str = "FREE" if self.state == BlockState.FREE else f"ALLOCATED(id={self.allocation_id})"
        return f"MemoryBlock(addr={self.address}, size={self.size}, state={state_str})"


@dataclass
class AllocationRecord:
    """
    Record of a successful allocation.
    
    Attributes:
        allocation_id: Unique identifier for this allocation.
        address: Starting address of the allocated block.
        requested_size: Original requested size.
        actual_size: Actual block size allocated (power of 2 >= requested_size).
        block: Reference to the allocated MemoryBlock.
    """
    allocation_id: int
    address: int
    requested_size: int
    actual_size: int
    block: MemoryBlock

    def __repr__(self) -> str:
        return (f"AllocationRecord(id={self.allocation_id}, addr={self.address}, "
                f"req={self.requested_size}, actual={self.actual_size})")


@dataclass
class AllocationResult:
    """
    Result of an allocation or deallocation operation.
    
    Attributes:
        status: SUCCESS or FAILURE.
        allocation_id: ID if successful, None if failed.
        address: Starting address if successful, None if failed.
        actual_size: Actual block size if successful, None if failed.
        message: Human-readable status message.
    """
    status: AllocationStatus
    allocation_id: Optional[int] = None
    address: Optional[int] = None
    actual_size: Optional[int] = None
    message: str = ""

    @property
    def success(self) -> bool:
        """Return True if the operation was successful."""
        return self.status == AllocationStatus.SUCCESS

    def __repr__(self) -> str:
        if self.success:
            return (f"AllocationResult(SUCCESS, id={self.allocation_id}, "
                    f"addr={self.address}, size={self.actual_size})")
        return f"AllocationResult(FAILURE, msg='{self.message}')"


@dataclass
class SystemStats:
    """
    Statistics about the buddy memory system.
    
    Attributes:
        total_memory: Total memory managed by the system.
        allocated_memory: Total bytes currently allocated.
        free_memory: Total bytes currently free.
        num_allocations: Number of active allocations.
        num_free_blocks: Number of free blocks in the free list.
        fragmentation_ratio: Ratio of allocated to total memory (0.0 to 1.0).
    """
    total_memory: int
    allocated_memory: int
    free_memory: int
    num_allocations: int
    num_free_blocks: int
    fragmentation_ratio: float

    def __repr__(self) -> str:
        return (f"SystemStats(total={self.total_memory}, allocated={self.allocated_memory}, "
                f"free={self.free_memory}, allocs={self.num_allocations}, "
                f"free_blocks={self.num_free_blocks}, frag={self.fragmentation_ratio:.4f})")
