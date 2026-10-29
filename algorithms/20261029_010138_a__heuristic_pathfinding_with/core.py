from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, List, Tuple, Optional

Coordinate = Tuple[int, int]
Grid = List[List[float]]
Heuristic = Callable[[Coordinate, Coordinate], float]


class AStar:
    """
    A* pathfinding on a mutable 2‑D cost grid.

    The grid stores traversal costs; a cell with cost math.inf is considered blocked.
    The algorithm supports dynamic updates to the grid between searches.
    """

    def __init__(
        self,
        cost_grid: Grid,
        heuristic: Optional[Heuristic] = None,
        allow_diagonal: bool = False,
    ) -> None:
        self.cost_grid = cost_grid
        self.rows = len(cost_grid)
        self.cols = len(cost_grid[0]) if self.rows else 0
        self.heuristic = heuristic or self.manhattan
        self.allow_diagonal = allow_diagonal

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def set_cost(self, coord: Coordinate, cost: float) -> None:
        """Update the traversal cost of a cell."""
        x, y = coord
        self._validate_coord(coord)
        self.cost_grid[y][x] = cost

    def find_path(self, start: Coordinate, goal: Coordinate) -> List[Coordinate]:
        """
        Compute the lowest‑cost path from start to goal.

        Returns a list of coordinates from start to goal inclusive.
        If no path exists, returns an empty list.
        """
        self._validate_coord(start)
        self._validate_coord(goal)

        if start == goal:
            return [start]

        open_set: List[Tuple[float, int, Coordinate]] = []
        heapq.heappush(open_set, (0.0, 0, start))

        came_from: Dict[Coordinate, Coordinate] = {}
        g_score: Dict[Coordinate, float] = {start: 0.0}
        f_score: Dict[Coordinate, float] = {start: self.heuristic(start, goal)}

        entry_counter = 1  # tie‑breaker for heap stability

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self._cost(neighbor)
                if tentative_g >= math.inf:
                    continue  # impassable

                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, entry_counter, neighbor))
                    entry_counter += 1

        # Exhausted search – no path found
        return []

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _cost(self, coord: Coordinate) -> float:
        x, y = coord
        return self.cost_grid[y][x]

    def _neighbors(self, coord: Coordinate) -> List[Coordinate]:
        x, y = coord
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.allow_diagonal:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        result: List[Coordinate] = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                if self._cost((nx, ny)) < math.inf:
                    result.append((nx, ny))
        return result

    @staticmethod
    def manhattan(a: Coordinate, b: Coordinate) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def euclidean(a: Coordinate, b: Coordinate) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _reconstruct_path(
        self, came_from: Dict[Coordinate, Coordinate], current: Coordinate
    ) -> List[Coordinate]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _validate_coord(self, coord: Coordinate) -> None:
        x, y = coord
        if not (0 <= x < self.cols and 0 <= y < self.rows):
            raise ValueError(f"Coordinate {coord} out of bounds for grid size {self.cols}x{self.rows}")
