from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from typing import List, Tuple, Callable


@dataclass(frozen=True)
class CostSegment:
    """A piecewise‑linear segment of a concave cost function.

    Attributes
    ----------
    max_quantity: Upper bound (inclusive) for which this segment is valid.
    slope: Marginal cost (must be non‑increasing across segments).
    intercept: Fixed cost component for this segment.
    """
    max_quantity: int
    slope: float
    intercept: float


class PiecewiseConcaveCost:
    """Evaluates a concave, piecewise‑linear production cost."""

    def __init__(self, segments: List[CostSegment]) -> None:
        if not segments:
            raise ValueError("At least one cost segment is required.")
        # Ensure segments are sorted by max_quantity and slopes are non‑increasing.
        sorted_segs = sorted(segments, key=lambda s: s.max_quantity)
        for i in range(1, len(sorted_segs)):
            if sorted_segs[i].slope > sorted_segs[i - 1].slope:
                raise ValueError("Slopes must be non‑increasing for concavity.")
        self._segments: List[CostSegment] = sorted_segs
        self._bounds: List[int] = [seg.max_quantity for seg in sorted_segs]

    def cost(self, quantity: int) -> float:
        """Return the production cost for a given total quantity."""
        if quantity < 0:
            raise ValueError("Quantity cannot be negative.")
        idx = bisect_left(self._bounds, quantity)
        if idx == len(self._segments):
            # Quantity exceeds all explicit bounds – use the last segment.
            seg = self._segments[-1]
        else:
            seg = self._segments[idx]
        return seg.slope * quantity + seg.intercept


def monotone_lot_sizing(
    demand: List[int],
    cost_func: PiecewiseConcaveCost,
) -> Tuple[List[float], List[int]]:
    """
    Compute the minimum total production cost for a series of periods
    using the monotone‑queue DP optimisation (valid for concave costs).

    Parameters
    ----------
    demand: List of demand quantities per period (non‑negative integers).
    cost_func: Callable that returns the cost for a given production quantity.

    Returns
    -------
    dp: List where dp[i] is the optimal cost to satisfy the first i periods.
    predecessor: List where predecessor[i] is the period index j (< i) that
                 yields the optimal split for dp[i].
    """
    n = len(demand)
    cum = [0] * (n + 1)
    for i in range(1, n + 1):
        cum[i] = cum[i - 1] + demand[i - 1]

    dp: List[float] = [0.0] * (n + 1)
    predecessor: List[int] = [0] * (n + 1)

    last_opt = 0  # monotone property: optimal j for i is non‑decreasing
    for i in range(1, n + 1):
        best_val = float("inf")
        best_j = -1
        # Search only from last_opt to i‑1 thanks to monotonicity.
        for j in range(last_opt, i):
            qty = cum[i] - cum[j]
            val = dp[j] + cost_func.cost(qty)
            if val < best_val:
                best_val = val
                best_j = j
        dp[i] = best_val
        predecessor[i] = best_j
        last_opt = best_j  # maintain monotone invariant

    return dp, predecessor


def reconstruct_schedule(
    predecessor: List[int],
    demand: List[int],
) -> List[Tuple[int, int, int]]:
    """
    Reconstruct the optimal production schedule from the predecessor array.

    Returns
    -------
    schedule: List of tuples (start_period, end_period, quantity_produced)
    """
    schedule: List[Tuple[int, int, int]] = []
    i = len(demand)
    while i > 0:
        j = predecessor[i]
        qty = sum(demand[j:i])
        schedule.append((j + 1, i, qty))  # periods are 1‑based for readability
        i = j
    schedule.reverse()
    return schedule
