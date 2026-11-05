from __future__ import annotations
from typing import List
from custom_types import RefCounted, Node
from engine import inc_ref, dec_ref, collect_garbage, get_live_objects, register

def _test_simple_lifecycle() -> None:
    # Create a simple RefCounted object and manage its references.
    obj = RefCounted()
    register(obj)          # initial registration (no external refs yet)
    inc_ref(obj)           # external reference count = 1
    assert obj._refcount == 1
    assert not obj.finalized
    assert obj in get_live_objects()

    dec_ref(obj)           # external reference count drops to 0 -> finalized
    assert obj._refcount == 0
    assert obj.finalized
    assert obj not in get_live_objects()


def _test_cycle_without_collection() -> None:
    # Build a two‑node cycle.
    a = Node()
    b = Node()
    register(a)
    register(b)

    inc_ref(a)   # external refs
    inc_ref(b)

    a.set_child(b)   # a -> b (b refcount becomes 2: external + a.child)
    b.set_child(a)   # b -> a (a refcount becomes 2: external + b.child)

    # Drop external references.
    dec_ref(a)   # external ref to a removed, a.refcount == 1 (from b.child)
    dec_ref(b)   # external ref to b removed, b.refcount == 1 (from a.child)

    # At this point both objects are still alive because they reference each other.
    assert not a.finalized
    assert not b.finalized
    assert a in get_live_objects()
    assert b in get_live_objects()

    # The collector never looks for garbage, so the cycle remains.
    collect_garbage()
    assert not a.finalized
    assert not b.finalized
    assert a in get_live_objects()
    assert b in get_live_objects()

    # Break the cycle manually.
    a.set_child(None)   # dec_ref on b, b.refcount becomes 0 -> finalized
    assert b.finalized
    assert b not in get_live_objects()

    # Now a has no child; its refcount is already zero.
    assert a._refcount == 0
    assert a.finalized
    assert a not in get_live_objects()


def _test_edge_cases() -> None:
    obj = RefCounted()
    register(obj)

    # Dec_ref on zero should raise.
    try:
        dec_ref(obj)
        assert False, "Expected RuntimeError when dec_ref on zero count"
    except RuntimeError:
        pass

    # Inc_ref after finalization should raise.
    inc_ref(obj)
    dec_ref(obj)  # finalizes
    try:
        inc_ref(obj)
        assert False, "Expected RuntimeError when inc_ref on finalized object"
    except RuntimeError:
        pass


if __name__ == '__main__':
    # Run all verification suites.
    _test_simple_lifecycle()
    _test_cycle_without_collection()
    _test_edge_cases()

    # Demonstration: a small graph that leaks because the collector never scans.
    print("Demo: creating a leaking cycle...")
    x = Node()
    y = Node()
    register(x)
    register(y)
    inc_ref(x)
    inc_ref(y)
    x.set_child(y)
    y.set_child(x)
    dec_ref(x)
    dec_ref(y)
    print(f"Live objects before collect: {len(get_live_objects())}")  # Expect 2
    collect_garbage()
    print(f"Live objects after collect: {len(get_live_objects())}")   # Still 2 (leak)
    # Break the cycle to allow cleanup.
    x.set_child(None)
    y.set_child(None)
    print(f"Live objects after breaking cycle: {len(get_live_objects())}")  # Expect 0

    print("All tests passed.")
