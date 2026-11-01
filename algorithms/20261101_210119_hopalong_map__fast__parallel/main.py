from __future__ import annotations

import random
import math
from typing import List

from .types import HopalongParams, Point, Orbit
from .engine import (
    generate_orbit,
    compute_checkpoints,
    recombine_segments,
)

def _orbit_to_lists(orbit: List[Point]) -> Tuple[List[float], List[float]]:
    """Utility for tests: split orbit into separate x and y lists."""
    xs = [p.x for p in orbit]
    ys = [p.y for p in orbit]
    return xs, ys

def _assert_orbits_equal(orbit1: List[Point], orbit2: List[Point], eps: float = 1e-12) -> None:
    """Assert two orbits are element‑wise equal within tolerance."""
    assert len(orbit1) == len(orbit2), f"Length mismatch {len(orbit1)} vs {len(orbit2)}"
    for i, (p1, p2) in enumerate(zip(orbit1, orbit2)):
        assert math.isclose(p1.x, p2.x, rel_tol=eps, abs_tol=eps), f"x mismatch at {i}"
        assert math.isclose(p1.y, p2.y, rel_tol=eps, abs_tol=eps), f"y mismatch at {i}"

def test_basic_properties() -> None:
    """Validate sign handling and single‑step correctness."""
    params = HopalongParams(a=0.0, b=0.0, c=0.0)
    init = Point(0.0, 0.0)
    # With all parameters zero, the map reduces to (0,0) forever
    orbit = list(generate_orbit(init, params, steps=5))
    for p in orbit:
        assert p.x == 0.0 and p.y == 0.0

def test_random_orbit_consistency() -> None:
    """Compare direct generation with checkpointed recombination."""
    random.seed(42)
    for _ in range(20):
        a = random.uniform(-2.0, 2.0)
        b = random.uniform(-2.0, 2.0)
        c = random.uniform(-2.0, 2.0)
        params = HopalongParams(a, b, c)
        init = Point(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
        total_steps = random.randint(0, 500)
        interval = random.randint(1, max(1, total_steps // 2))

        # Direct orbit
        direct_orbit = list(generate_orbit(init, params, total_steps))

        # Checkpointed recombination
        checkpoints = compute_checkpoints(init, params, total_steps, interval)
        recombined = recombine_segments(checkpoints, params, total_steps, interval)

        _assert_orbits_equal(direct_orbit, recombined)

def test_edge_cases() -> None:
    """Edge cases: zero steps, interval larger than total steps, negative inputs."""
    params = HopalongParams(1.0, 1.0, 1.0)
    init = Point(0.5, -0.5)

    # Zero steps
    orbit0 = list(generate_orbit(init, params, 0))
    assert len(orbit0) == 1 and orbit0[0] == init

    # Interval larger than total steps
    total = 7
    interval = 20
    checkpoints = compute_checkpoints(init, params, total, interval)
    assert len(checkpoints) == 2  # start and final
    recomb = recombine_segments(checkpoints, params, total, interval)
    direct = list(generate_orbit(init, params, total))
    _assert_orbits_equal(direct, recomb)

    # Invalid arguments raise
    try:
        compute_checkpoints(init, params, -1, 5)
        assert False, "negative total_steps should raise"
    except ValueError:
        pass
    try:
        compute_checkpoints(init, params, 10, 0)
        assert False, "non‑positive interval should raise"
    except ValueError:
        pass
    try:
        list(generate_orbit(init, params, -3))
        assert False, "negative steps should raise"
    except ValueError:
        pass

def demo() -> None:
    """Simple demonstration printing the first 10 points of a Hopalong orbit."""
    params = HopalongParams(a=0.9, b=-0.6013, c=2.0)
    init = Point(0.0, 0.0)
    steps = 10
    print("First 10 points of the Hopalong orbit (a=0.9, b=-0.6013, c=2.0):")
    for i, pt in enumerate(generate_orbit(init, params, steps)):
        print(f"{i:2d}: ({pt.x:.6f}, {pt.y:.6f})")

if __name__ == "__main__":
    # Run unit tests
    test_basic_properties()
    test_random_orbit_consistency()
    test_edge_cases()
    # Demonstration
    demo()
