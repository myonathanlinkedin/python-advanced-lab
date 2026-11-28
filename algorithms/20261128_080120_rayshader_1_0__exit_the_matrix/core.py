from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set


Direction = str  # Expected values: 'U', 'D', 'L', 'R'


@dataclass(frozen=True)
class SimulationResult:
    """
    Result of a ray traversal simulation.

    Attributes
    ----------
    exit_pos: Optional[Tuple[int, int]]
        The (row, col) coordinate where the ray left the matrix.
        None if the ray never exits (i.e., it loops forever).
    steps: int
        Number of steps taken before termination (including the step that exits).
    loop_detected: bool
        True if a loop was detected; False otherwise.
    """
    exit_pos: Optional[Tuple[int, int]]
    steps: int
    loop_detected: bool


def parse_matrix(raw: List[str]) -> List[List[Direction]]:
    """
    Convert a list of equal‑length strings into a matrix of direction characters.

    Parameters
    ----------
    raw: List[str]
        Each string represents a row; allowed characters are U, D, L, R.

    Returns
    -------
    List[List[Direction]]
        2‑D list representation.

    Raises
    ------
    ValueError
        If rows have inconsistent lengths or contain invalid characters.
    """
    if not raw:
        raise ValueError("Input matrix must contain at least one row.")
    width = len(raw[0])
    for row in raw:
        if len(row) != width:
            raise ValueError("All rows must have the same length.")
        for ch in row:
            if ch not in {"U", "D", "L", "R"}:
                raise ValueError(f"Invalid direction character: {ch!r}")
    return [list(row) for row in raw]


def _move(pos: Tuple[int, int], direction: Direction) -> Tuple[int, int]:
    """Return the next position after moving one step in the given direction."""
    r, c = pos
    if direction == "U":
        return r - 1, c
    if direction == "D":
        return r + 1, c
    if direction == "L":
        return r, c - 1
    if direction == "R":
        return r, c + 1
    raise ValueError(f"Unsupported direction: {direction!r}")


def simulate(matrix: List[List[Direction]],
             start: Tuple[int, int]) -> SimulationResult:
    """
    Simulate a ray moving through a direction matrix.

    The ray starts at `start`, reads the direction stored in that cell,
    moves one step accordingly, and repeats until it leaves the matrix
    or revisits a cell (loop detection).

    Parameters
    ----------
    matrix: List[List[Direction]]
        2‑D grid of direction characters.
    start: Tuple[int, int]
        Zero‑based (row, column) start coordinate.

    Returns
    -------
    SimulationResult
        Contains exit position (if any), step count, and loop flag.
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError("Start position out of matrix bounds.")

    visited: Set[Tuple[int, int]] = set()
    pos = start
    steps = 0

    while True:
        if pos in visited:
            # Loop detected
            return SimulationResult(exit_pos=None, steps=steps, loop_detected=True)
        visited.add(pos)
        direction = matrix[pos[0]][pos[1]]
        next_pos = _move(pos, direction)
        steps += 1
        # Check if next position is outside the matrix
        if not (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols):
            return SimulationResult(exit_pos=next_pos, steps=steps, loop_detected=False)
        pos = next_pos


def find_all_exits(matrix: List[List[Direction]]) -> List[SimulationResult]:
    """
    Run `simulate` for every cell as a start position.

    Returns a list of SimulationResult objects ordered row‑major.
    """
    results: List[SimulationResult] = []
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            results.append(simulate(matrix, (r, c)))
    return results
