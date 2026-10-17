import sys
from typing import List, Tuple
from custom_types import SimpleNamespace

# Import the R‑Tree implementation from the sibling modules.
from engine import RTree
from custom_types import Rectangle


def _point_rect(point: Tuple[float, ...]) -> Rectangle:
    """Convenient helper to create a zero‑area rectangle representing a point."""
    return Rectangle(point, point)


def _run_basic_tests() -> None:
    # Test 1: Empty tree search.
    tree = RTree(max_entries=4)
    query = Rectangle((0.0, 0.0), (1.0, 1.0))
    assert tree.search(query) == [], "Search on empty tree should return empty list."

    # Test 2: Insert a single point and query it.
    p = (0.5, 0.5)
    tree.insert(_point_rect(p), "A")
    result = tree.search(_point_rect(p))
    assert result == ["A"], "Single point insertion/search failed."

    # Test 3: Insert multiple points, some overlapping query region.
    points = [
        ((0.1, 0.1), "B"),
        ((0.2, 0.8), "C"),
        ((0.9, 0.9), "D"),
        ((0.4, 0.4), "E"),
        ((0.6, 0.2), "F"),
    ]
    for pt, label in points:
        tree.insert(_point_rect(pt), label)

    # Query a region that should intersect B, E, and A.
    query_rect = Rectangle((0.0, 0.0), (0.5, 0.5))
    found = set(tree.search(query_rect))
    expected = {"A", "B", "E"}
    assert found == expected, f"Range query returned {found}, expected {expected}"

    # Test 4: Force node splits by inserting many entries.
    many_points = [((i * 0.1, i * 0.1), f"P{i}") for i in range(20)]
    for pt, label in many_points:
        tree.insert(_point_rect(pt), label)

    # Verify that all inserted points are searchable.
    for pt, label in many_points:
        assert label in tree.search(_point_rect(pt)), f"Inserted point {label} not found."

    # Test 5: Verify that the tree size matches number of inserted objects.
    total_inserted = 1 + len(points) + len(many_points)  # A + points + many_points
    assert len(tree) == total_inserted, f"Tree length {len(tree)} != expected {total_inserted}"

    # Test 6: Query with a rectangle that does not intersect any entry.
    empty_query = Rectangle((10.0, 10.0), (20.0, 20.0))
    assert tree.search(empty_query) == [], "Non‑overlapping query should return empty list."

    print("All basic R‑Tree tests passed.")


def _run_edge_case_tests() -> None:
    # Edge case: Insert rectangles with zero area in one dimension (lines).
    tree = RTree(max_entries=3)
    line_rect = Rectangle((0.0, 0.0), (0.0, 5.0))  # vertical line at x=0
    tree.insert(line_rect, "Line")
    # Query a region that intersects the line.
    query = Rectangle((-1.0, 2.0), (1.0, 3.0))
    assert "Line" in tree.search(query), "Line rectangle not found in intersecting query."

    # Edge case: Identical rectangles.
    dup_rect = Rectangle((1.0, 1.0), (2.0, 2.0))
    tree.insert(dup_rect, "Dup1")
    tree.insert(dup_rect, "Dup2")
    result = set(tree.search(dup_rect))
    assert result == {"Dup1", "Dup2"}, "Identical rectangles handling failed."

    # Edge case: Very high dimensionality (5‑D).
    high_dim_rect = Rectangle((0, 0, 0, 0, 0), (1, 1, 1, 1, 1))
    tree.insert(high_dim_rect, "5D")
    query5d = Rectangle((0.5, 0.5, 0.5, 0.5, 0.5), (2, 2, 2, 2, 2))
    assert "5D" in tree.search(query5d), "5‑D rectangle not found."

    print("All edge‑case R‑Tree tests passed.")


if __name__ == "__main__":
    _run_basic_tests()
    _run_edge_case_tests
