import unittest
from core import AStar
from typing import Tuple, List, Optional

class TestAStar(unittest.TestCase):
    def test_simple_grid(self):
        start = (0, 0)
        goal = (4, 4)
        result = AStar.find_path(start, goal)
        self.assertIsNotNone(result)
        path, cost = result
        self.assertEqual(cost, 8.0)
        self.assertEqual(len(path), 9)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)

    def test_obstacle_grid(self):
        obstacles = {(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)}
        def cost_func(pos: Tuple[int, int]) -> float:
            return 100.0 if pos in obstacles else 1.0
        start = (0, 0)
        goal = (4, 4)
        result = AStar.find_path(start, goal, cost_func=cost_func, grid_size=(5, 5))
        self.assertIsNotNone(result)
        path, cost = result
        self.assertGreater(cost, 8.0)
        self.assertNotIn((1, 2), path)

    def test_unreachable(self):
        obstacles = {(0, 1), (0, 2), (0, 3), (0, 4)}
        def cost_func(pos: Tuple[int, int]) -> float:
            return 100.0 if pos in obstacles else 1.0
        start = (0, 0)
        goal = (0, 4)
        result = AStar.find_path(start, goal, cost_func=cost_func, grid_size=(5, 5))
        self.assertIsNone(result)

    def test_dynamic_cost(self):
        dynamic_state = {'blocked': {(2, 2)}}
        def cost_func(pos: Tuple[int, int]) -> float:
            return 100.0 if pos in dynamic_state['blocked'] else 1.0
        start = (0, 0)
        goal = (4, 4)
        # Block surrounding cells to force path through (2,2)
        surrounding = {(1, 2), (2, 1), (2, 3), (3, 2)}
        dynamic_state['blocked'].update(surrounding)
        result1 = AStar.find_path(start, goal, cost_func=cost_func, grid_size=(5, 5))
        self.assertIsNotNone(result1)
        path1, cost1 = result1
        self.assertIn((2, 2), path1)
        # Remove the block at (2,2) and recompute
        dynamic_state['blocked'].remove((2, 2))
        result2 = AStar.find_path(start, goal, cost_func=cost_func, grid_size=(5, 5))
        self.assertIsNotNone(result2)
        path2, cost2 = result2
        self.assertNotIn((2, 2), path2)
        self.assertLess(cost2, cost1)

    def test_grid_bounds(self):
        start = (0, 0)
        goal = (2, 2)
        result = AStar.find_path(start, goal, grid_size=(3, 3))
        self.assertIsNotNone(result)
        path, cost = result
        for pos in path:
            self.assertTrue(0 <= pos[0] < 3 and 0 <= pos[1] < 3)

if __name__ == '__main__':
    unittest.main()
