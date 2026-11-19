from custom_types import BoundingBox, BoxEntry
from engine import StratumTreeIndex

def _test_basic_2d() -> None:
    idx = StratumTreeIndex(dimension=2, max_entries=4)

    # Insert a few boxes
    idx.insert("A", (0, 0), (2, 2))
    idx.insert("B", (1, 1), (3, 3))
    idx.insert("C", (5, 5), (6, 6))
    idx.insert("D", (2, 0), (4, 1))

    # Query overlapping region
    result = set(idx.query((1, 1), (2.5, 2.5)))
    assert result == {"A", "B"}, f"Unexpected 2D query result: {result}"

    # Query non‑overlapping region
    result = set(idx.query((4, 4), (5, 5)))
    assert result == set(), f"Expected empty result, got {result}"

    # Query that touches edge
    result = set(idx.query((4, 0), (5, 1)))
    assert result == {"D"}, f"Edge‑touch query failed: {result}"

def _test_basic_3d() -> None:
    idx = StratumTreeIndex(dimension=3, max_entries=3)

    idx.insert(1, (0, 0, 0), (1, 1, 1))
    idx.insert(2, (0.5, 0.5, 0.5), (2, 2, 2))
    idx.insert(3, (3, 3, 3), (4, 4, 4))

    # Overlap query
    result = set(idx.query((0.75, 0.75, 0.75), (1.5, 1.5, 1.5)))
    assert result == {1, 2}, f"3D overlap query failed: {result}"

    # Exact match query
    result = set(idx.query((3, 3, 3), (4, 4, 4)))
    assert result == {3}, f"Exact match query failed: {result}"

def _test_edge_cases() -> None:
    # Zero‑volume boxes (points)
    idx = StratumTreeIndex(dimension=2, max_entries=2)
    idx.insert("P1", (1, 1), (1, 1))
    idx.insert("P2", (2, 2), (2, 2))

    # Query a point that matches one entry
    result = set(idx.query((1, 1), (1, 1)))
    assert result == {"P1"}, f"Point query failed: {result}"

    # Query a region that includes both points
    result = set(idx.query((0, 0), (3, 3)))
    assert result == {"P1", "P2"}, f"Region covering points failed: {result}"

    # Empty tree query
    empty_idx = StratumTreeIndex(dimension=2)
    result = empty_idx.query((0, 0), (1, 1))
    assert result == [], f"Empty tree should return empty list, got {result}"

def _test_tree_growth() -> None:
    # Force multiple splits and height increase
    idx = StratumTreeIndex(dimension=2, max_entries=3)
    for i in range(10):
        idx.insert(i, (i, i), (i + 0.5, i + 0.5))

    # Verify that all inserted ids are retrievable
    all_ids = set(idx.query((-1, -1), (20, 20)))
    expected = set(range(10))
    assert all_ids == expected, f"Tree growth test failed, missing ids: {expected - all_ids}"

if __name__ == "__main__":
    _test_basic_2d()
    _test_basic_3d()
    _test_edge_cases()
    _test_tree_growth()
    print("All StratumTreeIndex unit tests passed.")
