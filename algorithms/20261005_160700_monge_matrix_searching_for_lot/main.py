import random
import time
import unittest
from typing import List, Tuple

from core import CostSegment, PiecewiseConcaveCost, monotone_lot_sizing, reconstruct_schedule


def demo() -> None:
    """Simple demonstration of the lot‑sizing algorithm."""
    demand = [10, 20, 15, 5, 25]
    # Concave cost: first 30 units at $5/unit, thereafter $3/unit.
    segments = [
        CostSegment(max_quantity=30, slope=5.0, intercept=0.0),
        CostSegment(max_quantity=10**9, slope=3.0, intercept=0.0),
    ]
    cost_func = PiecewiseConcaveCost(segments)
    dp, pred = monotone_lot_sizing(demand, cost_func)
    schedule = reconstruct_schedule(pred, demand)

    print("Optimal total cost:", dp[-1])
    print("Production schedule (period start‑end, quantity):")
    for s in schedule:
        print(s)


class TestLotSizing(unittest.TestCase):
    def test_basic_concave(self) -> None:
        demand = [10, 10, 10]
        segments = [
            CostSegment(max_quantity=15, slope=6.0, intercept=0.0),
            CostSegment(max_quantity=10**9, slope=4.0, intercept=0.0),
        ]
        cost_func = PiecewiseConcaveCost(segments)
        dp, pred = monotone_lot_sizing(demand, cost_func)
        # Manual enumeration:
        # Produce all 30 at once: cost = 6*15 + 4*15 = 150
        # Produce first 20, then 10: cost = 6*15 + 4*5 + 4*10 = 140
        # Produce each period separately: 3 * (6*10) = 180
        self.assertAlmostEqual(dp[-1], 140.0)
        schedule = reconstruct_schedule(pred, demand)
        self.assertEqual(schedule, [(1, 2, 20), (3, 3, 10)])

    def test_monotone_property(self) -> None:
        # Random demand with a strongly concave cost should keep opt indices monotone.
        demand = [random.randint(1, 20) for _ in range(50)]
        segments = [
            CostSegment(max_quantity=40, slope=8.0, intercept=0.0),
            CostSegment(max_quantity=10**9, slope=2.0, intercept=0.0),
        ]
        cost_func = PiecewiseConcaveCost(segments)
        dp, pred = monotone_lot_sizing(demand, cost_func)
        # Verify monotonicity of predecessor indices.
        for i in range(2, len(pred)):
            self.assertGreaterEqual(pred[i], pred[i - 1])

    def test_invalid_segments(self) -> None:
        with self.assertRaises(ValueError):
            PiecewiseConcaveCost([])  # no segments

        # Slopes increasing violates concavity.
        segs = [
            CostSegment(max_quantity=10, slope=5.0, intercept=0.0),
            CostSegment(max_quantity=20, slope=6.0, intercept=0.0),
        ]
        with self.assertRaises(ValueError):
            PiecewiseConcaveCost(segs)

    def test_zero_demand(self) -> None:
        demand: List[int] = []
        segments = [CostSegment(max_quantity=10**9, slope=1.0, intercept=0.0)]
        cost_func = PiecewiseConcaveCost(segments)
        dp, pred = monotone_lot_sizing(demand, cost_func)
        self.assertEqual(dp, [0.0])
        self.assertEqual(pred, [0])

    def test_large_instance_performance(self) -> None:
        n = 5000
        demand = [random.randint(1, 100) for _ in range(n)]
        segments = [
            CostSegment(max_quantity=2000, slope=7.0, intercept=0.0),
            CostSegment(max_quantity=10**9, slope=3.0, intercept=0.0),
        ]
        cost_func = PiecewiseConcaveCost(segments)
        start = time.perf_counter()
        dp, _ = monotone_lot_sizing(demand, cost_func)
        duration = time.perf_counter() - start
        # Ensure the algorithm runs in sub‑second for n=5000 on typical hardware.
        self.assertLess(duration, 1.0)
        self.assertIsInstance(dp[-1], float)


if __name__ == "__main__":
    # Run unit tests; if they pass, show a demo.
    unittest.main(exit=False)
    print("\n--- Demo Run ---")
    demo()
