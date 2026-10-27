from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import Deque, List, Optional


class ProcessState(Enum):
    READY = auto()
    RUNNING = auto()
    BLOCKED = auto()
    TERMINATED = auto()


@dataclass
class Process:
    pid: int
    total_time: int  # total CPU time required
    remaining_time: int = field(init=False)
    state: ProcessState = field(default=ProcessState.READY, init=False)

    def __post_init__(self) -> None:
        if self.total_time < 0:
            raise ValueError("total_time must be non‑negative")
        self.remaining_time = self.total_time

    def is_finished(self) -> bool:
        return self.remaining_time <= 0


class Scheduler:
    """
    Simple round‑robin scheduler with a fixed time quantum.
    """

    def __init__(self, quantum: int) -> None:
        if quantum <= 0:
            raise ValueError("quantum must be positive")
        self.quantum: int = quantum
        self._ready: Deque[Process] = deque()
        self._current: Optional[Process] = None
        self._quantum_counter: int = 0

    def add_process(self, proc: Process) -> None:
        if proc.state != ProcessState.READY:
            raise ValueError("Only READY processes can be added")
        self._ready.append(proc)

    def tick(self) -> Optional[Process]:
        """
        Simulate one timer tick.
        Returns the process that was running during this tick,
        or None if the system is idle.
        """
        # Load a new process if CPU is idle
        if self._current is None:
            if not self._ready:
                return None
            self._current = self._ready.popleft()
            self._current.state = ProcessState.RUNNING
            self._quantum_counter = self.quantum

        proc = self._current
        # Execute one unit of time
        proc.remaining_time -= 1
        self._quantum_counter -= 1

        # Check for termination
        if proc.is_finished():
            proc.state = ProcessState.TERMINATED
            self._current = None
            return proc

        # Check for quantum expiration (preempt)
        if self._quantum_counter == 0:
            proc.state = ProcessState.READY
            self._ready.append(proc)
            self._current = None

        return proc

    def is_idle(self) -> bool:
        return self._current is None and not self._ready

    def __len__(self) -> int:
        """Number of processes that are not terminated."""
        count = len(self._ready)
        if self._current and self._current.state != ProcessState.TERMINATED:
            count += 1
        return count


class MemoryManager:
    """
    Very small bitmap‑based page allocator.
    Pages are identified by integer indices [0, total_pages).
    """

    def __init__(self, total_pages: int) -> None:
        if total_pages <= 0:
            raise ValueError("total_pages must be positive")
        self.total_pages: int = total_pages
        self._bitmap: List[bool] = [False] * total_pages  # False == free

    def allocate(self, num_pages: int) -> List[int]:
        """
        Allocate any free pages (not required to be contiguous).
        Returns a list of allocated page indices.
        Raises MemoryError if insufficient free pages exist.
        """
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        free_indices = [i for i, used in enumerate(self._bitmap) if not used]
        if len(free_indices) < num_pages:
            raise MemoryError("Not enough free pages")
        allocated = free_indices[:num_pages]
        for i in allocated:
            self._bitmap[i] = True
        return allocated

    def free(self, pages: List[int]) -> None:
        """
        Release previously allocated pages.
        Raises ValueError if a page is already free or out of range.
        """
        for i in pages:
            if not (0 <= i < self.total_pages):
                raise ValueError(f"Page index {i} out of range")
            if not self._bitmap[i]:
                raise ValueError(f"Page {i} is already free")
            self._bitmap[i] = False

    def used_pages(self) -> List[int]:
        """Return a list of currently allocated page indices."""
        return [i for i, used in enumerate(self._bitmap) if used]

    def free_pages(self) -> List[int]:
        """Return a list of currently free page indices."""
        return [i for i, used in enumerate(self._bitmap) if not used]
