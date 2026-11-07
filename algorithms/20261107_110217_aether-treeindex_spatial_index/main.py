from __future__ import annotations
from typing import List, Tuple
from custom_types import BoundingBox
from engine import AetherTreeIndex


def _make_bbox(mins: Tuple[float, ...], maxs: Tuple[float, ...]) -> BoundingBox:
    return BoundingBox(mins, maxs)


def _basic_insertion_and_query_test() -> None:
    tree = AetherTreeIndex(dimensions=2, max_entries=4)

    # Insert a few boxes with identifiable payloads
    boxes = [
        (_make_bbox((0, 0), (2, 2)), "A"),
        (_make_bbox((1, 1), (3, 3)), "B"),
        (_make_bbox((5, 5), (7, 7)), "C"),
        (_make_bbox((6, 0), (8, 2)), "D"),
        (_make_bbox((2, 2), (4, 4)), "E"),  # forces a split
    ]

    for bbox, payload in boxes:
        tree.insert(bbox, payload)

    # Query that overlaps A, B, and E
    query = _make_bbox((1.5, 1.5), (3.5, 3.5))
    hits = set(tree.search(query))
    assert hits == {"A", "B", "E"}, f"Unexpected query result: {hits}"

    # Query that overlaps only C
    query2 = _make_bbox((5.5, 5.5), (6.5, 6.5))
    hits2 = set(tree.search(query2))
    assert hits2 == {"C"}, f"Unexpected query result: {hits2}"

    # Query with no overlap
    query3 = _make_bbox((9, 9), (10, 10))
    hits3 = tree.search(query3)
    assert hits3 == [], f"Expected empty result, got {hits3}"


def _edge_cases_test() -> None:
    # Zero‑area boxes (points) and degenerate dimensions
    tree = AetherTreeIndex(dimensions=3, max_entries=3)

    point = _make_bbox((1, 1, 1), (1, 1, 1))
    line = _make_bbox((0, 0, 0), (5, 0, 0))
    cube = _make_bbox((-1, -1, -1), (2, 2, 2))

    tree.insert(point, "point")
    tree.insert(line, "line")
    tree.insert(cube, "cube")

    # Query that hits all three
    q_all = _make_bbox((0, 0, 0), (1, 1, 1))
    assert set(tree.search(q_all)) == {"point", "line", "cube"}

    # Query that hits only the line
    q_line = _make_bbox((-1, -0.1, -0.1), (6, 0.1, 0.1))
    assert set(tree.search(q_line)) == {"line", "cube"}

    # Query that hits nothing
    q_none = _make_bbox((10, 10, 10), (12, 12, 12))
    assert tree.search(q_none) == []


def _large_random_test() -> None:
    import random

    random.seed(0)
    dim = 2
    tree = AetherTreeIndex(dimensions=dim, max_entries=8)

    # Insert 500 random boxes
    payloads = []
    for i in range(500):
        x1 = random.uniform(0, 1000)
        y1 = random.uniform(0, 1000)
        w = random.uniform(0.1, 20)
        h = random.uniform(0.1, 20)
        bbox = _make_bbox((x1, y1), (x1 + w, y1 + h))
        payload = f"id_{i}"
        payloads.append((bbox, payload))
        tree.insert(bbox, payload)

    # Perform 20 random queries and cross‑validate with brute force
    for _ in range(20):
        qx = random.uniform(0, 1000)
        qy = random.uniform(0, 1000)
        qw = random.uniform(0, 30)
        qh = random.uniform(0, 30)
        query = _make_bbox((qx, qy), (qx + qw, qy + qh))
        expected = {payload for bbox, payload in payloads if bbox.intersects(query)}
        result = set(tree.search(query))
        assert result == expected, f"Mismatch in random query: expected {len(expected)} got {len(result)}"


if __name__ == "__main__":
    _basic_insertion_and_query_test()
    _edge_cases_test()
    _large_random_test()
    print("All AetherTreeIndex unit tests passed.")
