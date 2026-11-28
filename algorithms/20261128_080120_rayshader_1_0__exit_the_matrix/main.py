import unittest
from typing import List, Tuple

from core import (
    parse_matrix,
    simulate,
    find_all_exits,
    SimulationResult,
)


class TestRayshader(unittest.TestCase):
    def test_parse_matrix_valid(self):
        raw = ["RDL", "UUR"]
        mat = parse_matrix(raw)
        self.assertEqual(mat, [['R', 'D', 'L'], ['U', 'U', 'R']])

    def test_parse_matrix_invalid_char(self):
        with self.assertRaises(ValueError):
            parse_matrix(["AB"])

    def test_parse_matrix_inconsistent_rows(self):
        with self.assertRaises(ValueError):
            parse_matrix(["R", "LR"])

    def test_simulate_exit_immediate(self):
        mat = parse_matrix(["R"])
        res = simulate(mat, (0, 0))
        self.assertEqual(res.exit_pos, (0, 1))
        self.assertFalse(res.loop_detected)
        self.assertEqual(res.steps, 1)

    def test_simulate_loop(self):
        mat = parse_matrix(["R", "L"])
        # Matrix:
        # R
        # L
        # Starting at (0,0) goes right (out), but we test a loop:
        mat_loop = parse_matrix(["R", "U"])
        # (0,0) -> (0,1) out of bounds, not a loop. Build a true loop:
        mat_loop = parse_matrix(["R", "L"])
        # Actually need a 2x2 loop:
        mat_loop = parse_matrix(["RD", "LU"])
        res = simulate(mat_loop, (0, 0))
        self.assertTrue(res.loop_detected)
        self.assertIsNone(res.exit_pos)

    def test_simulate_complex(self):
        raw = [
            "RRD",
            "ULL",
            "UUR"
        ]
        mat = parse_matrix(raw)
        # Start at (0,0): R -> (0,1) R -> (0,2) D -> (1,2) L -> (1,1) L -> (1,0) U -> (0,0) loop
        res = simulate(mat, (0, 0))
        self.assertTrue(res.loop_detected)
        self.assertIsNone(res.exit_pos)

        # Start at (2,2): R -> out
        res2 = simulate(mat, (2, 2))
        self.assertFalse(res2.loop_detected)
        self.assertEqual(res2.exit_pos, (2, 3))

    def test_find_all_exits(self):
        raw = [
            "RR",
            "LL"
        ]
        mat = parse_matrix(raw)
        results = find_all_exits(mat)
        self.assertEqual(len(results), 4)
        # (0,0) -> (0,1) -> out
        self.assertFalse(results[0].loop_detected)
        self.assertEqual(results[0].exit_pos, (0, 2))
        # (0,1) -> out immediately
        self.assertFalse(results[1].loop_detected)
        self.assertEqual(results[1].exit_pos, (0, 2))
        # (1,0) -> (1,-1) out left
        self.assertFalse(results[2].loop_detected)
        self.assertEqual(results[2].exit_pos, (1, -1))
        # (1,1) -> out left
        self.assertFalse(results[3].loop_detected)
        self.assertEqual(results[3].exit_pos, (1, -1))

    def test_simulate_start_out_of_bounds(self):
        mat = parse_matrix(["R"])
        with self.assertRaises(ValueError):
            simulate(mat, (1, 0))

    def test_large_matrix_performance(self):
        # 50x50 matrix where every cell points right; should exit after 1 step from any cell.
        size = 50
        raw = ["R" * size for _ in range(size)]
        mat = parse_matrix(raw)
        results = find_all_exits(mat)
        for res in results:
            self.assertFalse(res.loop_detected)
            # column index = size-1, moving right exits at col=size
            self.assertEqual(res.exit_pos, (res.exit_pos[0], size))


def demo() -> None:
    """
    Simple demonstration of the Rayshader algorithm.
    """
    raw = [
        "RRD",
        "ULL",
        "UUR"
    ]
    matrix = parse_matrix(raw)
    start = (0, 0)
    result = simulate(matrix, start)
    print(f"Start at {start}:")
    if result.loop_detected:
        print(f"  Detected a loop after {result.steps} steps.")
    else:
        print(f"  Exited at {result.exit_pos} after {result.steps} steps.")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n--- Demo ---")
    demo()
