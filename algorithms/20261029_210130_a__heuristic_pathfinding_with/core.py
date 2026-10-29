from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

Position = Tuple[int, int]
CostFunc = Callable[[Position], float]
HeuristicFunc = Callable[[Position, Position], float]

@dataclass(order=True)
class Node:
    f: float
    position: Position = field(compare=False)
    g: float = field(compare=False)
    h: float = field(compare=False)
    parent: Optional[Node] = field(compare=False, default=None)

class AStar:
    @staticmethod
    def find_path(
        start: Position,
        goal: Position,
        cost_func: CostFunc = lambda pos: 1.0,
        heuristic_func: HeuristicFunc = lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1]),
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[List[Position], float]]:
        """
        Find a path from start to goal using the A* algorithm.
        Returns a tuple (path, total_cost) or None if no path exists.
        """
        open_set: List[Node] = []
        heapq.heappush(open_set, Node(0.0, start, 0.0, 0.0, None))
        g_scores: Dict[Position, float] = {start: 0.0}
        closed_set: Set[Position] = set()

        while open_set:
            current = heapq.heappop(open_set)
            if current.position == goal:
                path: List[Position] = []
                node = current
                while node:
                    path.append(node.position)
                    node = node.parent
                path.reverse()
                return path, current.g

            closed_set.add(current.position)

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current.position[0] + dx, current.position[1] + dy)
                if grid_size:
                    if not (0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]):
                        continue
                if neighbor in closed_set:
                    continue
                tentative_g = current.g + cost_func(neighbor)
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = tentative_g
                    h = heuristic_func(neighbor, goal)
                    f = tentative_g + h
                    heapq.heappush(open_set, Node(f, neighbor, tentative_g, h, current))
        return None
