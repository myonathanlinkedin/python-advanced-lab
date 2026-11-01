from __future__ import annotations

from typing import Dict

from custom_types import ComparisonResult, ProcessId
from engine import VectorClock


def _assert_equal(a: VectorClock, b: VectorClock) -> None:
    assert a == b, f"Expected equality: {a!r} != {b!r}"
    assert a.compare(b) == ComparisonResult.EQUAL, f"Comparison mismatch: {a!r} vs {b!r}"


def _assert_less(a: VectorClock, b: VectorClock) -> None:
    assert a.compare(b) == ComparisonResult.LESS, f"Expected {a!r} < {b!r}"
    assert b.compare(a) == ComparisonResult.GREATER, f"Expected {b!r} > {a!r}"


def _assert_concurrent(a: VectorClock, b: VectorClock) -> None:
    assert a.compare(b) == ComparisonResult.CONCURRENT, f"Expected concurrency: {a!r} || {b!r}"
    assert b.compare(a) == ComparisonResult.CONCURRENT, f"Expected concurrency: {b!r} || {a!r}"


def run_unit_tests() -> None:
    # Basic construction and increment
    p1: ProcessId = "A"
    p2: ProcessId = "B"
    p3: ProcessId = "C"

    vc_a = VectorClock(p1)
    vc_b = VectorClock(p2)
    vc_c = VectorClock(p3)

    # Initially all clocks are equal to zero for their own pid
    _assert_equal(vc_a, VectorClock(p1, {p1: 0}))
    _assert_equal(vc_b, VectorClock(p2, {p2: 0}))
    _assert_equal(vc_c, VectorClock(p3, {p3: 0}))

    # Increment A twice, B once
    vc_a.increment()
    vc_a.increment()
    vc_b.increment()

    assert vc_a.clock[p1] == 2
    assert vc_b.clock[p2] == 1

    # Compare distinct clocks (should be concurrent)
    _assert_concurrent(vc_a, vc_b)

    # Merge B into A (join)
    vc_a.update(vc_b)
    # After merge, A knows about B's event
    assert vc_a.clock[p1] == 2
    assert vc_a.clock[p2] == 1

    # Now A dominates B
    _assert_greater(vc_a, vc_b)

    # Increment B to catch up
    vc_b.increment()  # B now at 2
    vc_b.increment()  # B now at 3
    # B's clock: {B:3}
    # Merge A into B
    vc_b.update(vc_a)
    # B should now have A:2, B:3
    assert vc_b.clock[p1] == 2
    assert vc_b.clock[p2] == 3

    # A is now less than B
    _assert_less(vc_a, vc_b)

    # Test serialization round‑trip
    serialized: Dict[ProcessId, int] = vc_b.to_dict()
    restored = VectorClock.from_dict(p2, serialized)
    _assert_equal(vc_b, restored)

    # Edge case: missing entries are treated as zero
    vc_sparse = VectorClock(p1, {p2: 5})  # local pid A not present yet
    assert vc_sparse.clock[p1] == 0
    assert vc_sparse.clock[p2] == 5
    vc_sparse.increment()
    assert vc_sparse.clock[p1] == 1

    # Compare sparse with a full clock
    full = VectorClock(p2, {p1: 1, p2: 5})
    _assert_equal(vc_sparse, full)

    # Concurrency detection with three processes
    vc1 = VectorClock("X")
    vc2 = VectorClock("Y")
    vc3 = VectorClock("Z")

    vc1.increment()  # X:1
    vc2.increment()  # Y:1
    vc3.increment()  # Z:1

    _assert_concurrent(vc1, vc2)
    _assert_concurrent(vc1, vc3)
    _assert_concurrent(vc2, vc3)

    # Merge all into vc1
    vc1.update(vc2)
    vc1.update(vc3)
    # Now vc1 dominates the others
    _assert_greater(vc1, vc2)
    _assert_greater(vc1, vc3)

    # Verify that copy produces an independent object
    vc_copy = vc1.copy()
    assert vc_copy is not vc1
    assert vc_copy == vc1
    vc_copy.increment()
    assert vc_copy != vc1  # mutation of copy must not affect original

    print("All unit tests passed.")


def _assert_greater(a: VectorClock, b: VectorClock) -> None:
    assert a.compare(b) == ComparisonResult.GREATER, f"Expected {a!r} > {b!r}"
    assert b.compare(a) == ComparisonResult.LESS, f"Expected {b!r} < {a!r}"


if __name__ == "__main__":
    run_unit_tests()
