import unittest
import math
from typing import List, Tuple

from core import AStar, Grid, Coordinate


def make_grid(width: int, height: int, default_cost: float = 1.0) -> Grid:
    """Utility to create a rectangular grid filled with default_cost."""
    return [[default_cost for _ in range(width)] for _ in range(height)]


class TestAStar(unittest.TestCase):
    def test_straight_line(self) -> None:
        grid = make_grid(5, 5)
        astar = AStar(grid)
        start = (0, 0)
        goal = (4, 0)
        path = astar.find_path(start, goal)
        expected = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        self.assertEqual(path, expected)

    def test_obstacle_avoidance(self) -> None:
        grid = make_grid(5, 5)
        # Insert a vertical wall blocking the direct route
        for y in range(5):
            grid[y][2] = math.inf
        # Open a gap at (2,2)
        grid[2][2] = 1.0

        astar = AStar(grid)
        start = (0, 2)
        goal = (4, 2)
        path = astar.find_path(start, goal)

        # Path must go through the gap at (2,2)
        self.assertIn((2, 2), path)
        self.assertNotIn((2, 0), path)  # ensure it didn't cut through blocked cells

    def test_dynamic_cost_update(self) -> None:
        grid = make_grid(3, 3)
        astar = AStar(grid)

        start = (0, 0)
        goal = (2, 2)

        # Initially all costs are equal – diagonal path is optimal
        path1 = astar.find_path(start, goal)
        self.assertEqual(path1, [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)])

        # Increase cost of the rightmost column dramatically
        for y in range(3):
            astar.set_cost((2, y), 100.0)

        path2 = astar.find_path(start, goal)
        # Now the algorithm should prefer the bottom row
        self.assertEqual(path2, [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])

    def test_unreachable_goal(self) -> None:
        grid = make_grid(3, 3)
        # Surround the goal with impassable cells
        grid[1][1] = math.inf
        grid[0][1] = math.inf
        grid[1][0] = math.inf
        grid[2][1] = math.inf
        grid[1][2] = math.inf

        astar = AStar(grid)
        start = (0, 0)
        goal = (1, 1)
        path = astar.find_path(start, goal)
        self.assertEqual(path, [])  # No possible path

    def test_start_equals_goal(self) -> None:
        grid = make_grid(2, 2)
        astar = AStar(grid)
        start = (1, 1)
        goal = (1, 1)
        path = astar.find_path(start, goal)
        self.assertEqual(path, [(1, 1)])

    def test_diagonal_movement(self) -> None:
        grid = make_grid(3, 3)
        astar = AStar(grid, allow_diagonal=True)
        start = (0, 0)
        goal = (2, 2)
        path = astar.find_path(start, goal)
        # With diagonal moves allowed, the optimal path length is 3 steps
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)


def demo() -> None:
    """Simple interactive demonstration of A* on a 7×7 grid."""
    width, height = 7, 7
    grid = make_grid(width, height)
    # Create a cross‑shaped obstacle with higher cost
    for i in range(width):
        grid[3][i] = 5.0  # horizontal bar
        grid[i][3] = 5.0  # vertical bar
    # Open a passage
    grid[3][3] = 1.0

    astar = AStar(grid, allow_diagonal=False)
    start = (0, 0)
    goal = (6, 6)
    path = astar.find_path(start, goal)

    # Render the grid with the path
    for y in range(height):
        row = ""
        for x in range(width):
            if (x, y) == start:
                row += "S "
            elif (x, y) == goal:
                row += "G "
            elif (x, y) in path:
                row += "* "
            elif grid[y][x] >= math.inf:
                row += "X "
            elif grid[y][x] > 1.0:
                row += "~ "
            else:
                row += ". "
        print(row.rstrip())
    print(f"\nPath length: {len(path)} steps")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a quick demo
    print("\n--- Demo ---")
    demo()
