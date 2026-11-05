from __future__ import annotations
from typing import List, Tuple
from custom_types import Rectangle
from engine import RTree


def _make_rect(min_coords: Tuple[float, ...], max_coords: Tuple[float, ...]) -> Rectangle:
    return Rectangle(min_coords, max_coords)


def _basic_insertion_test() -> None:
    tree = RTree(max_entries=4)
    data = [
        (1, _make_rect((0, 0), (1, 1))),
        (2, _make_rect((2, 2), (3, 3))),
        (3, _make_rect((1, 1), (2, 2))),
        (4, _make_rect((3, 0), (4, 1))),
        (5, _make_rect((0, 3), (1, 4))),
    ]
    for item, rect in data:
        tree.insert(item, rect)

    # Query that should intersect items 1,3
    q = _make_rect((0.5, 0.5), (1.5, 1.5))
    hits = set(tree.search(q))
    assert hits == {1, 3}, f"Expected {{1,3}} got {hits}"

    # Query that intersects nothing
    q2 = _make_rect((5, 5), (6, 6))
    hits2 = tree.search(q2)
    assert hits2 == [], f"Expected empty list, got {hits2}"

    # Query that intersects multiple items
    q3 = _make_rect((0, 0), (4, 4))
    hits3 = set(tree.search(q3))
    assert hits3 == {1, 2, 3, 4, 5}, f"All items should be returned, got {hits3}"


def _empty_tree_test() -> None:
    tree = RTree()
    q = _make_rect((0, 0), (1, 1))
    assert tree.search(q) == [], "Search on empty tree must return empty list"


def _duplicate_rectangles_test() -> None:
    tree = RTree(max_entries=3)
    rect = _make_rect((0, 0), (2, 2))
    for i in range(5):
        tree.insert(f"obj{i}", rect)

    # All objects should be found
    hits = set(tree.search(rect))
    expected = {f"obj{i}" for i in range(5)}
    assert hits == expected, f"All duplicate objects should be returned, got {hits}"


def _high_dimensional_test() -> None:
    # 3‑D example
    tree = RTree(max_entries=4)
    data = [
        ("a", _make_rect((0, 0, 0), (1, 1, 1))),
        ("b", _make_rect((2, 2, 2), (3, 3, 3))),
        ("c", _make_rect((1, 1, 1), (2, 2, 2))),
    ]
    for item, rect in data:
        tree.insert(item, rect)

    q = _make_rect((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
    hits = set(tree.search(q))
    assert hits == {"a", "c"}, f"3‑D query failed, got {hits}"


def _run_all_tests() -> None:
    _empty_tree_test()
    _basic_insertion_test()
    _duplicate_rectangles_test()
    _high_dimensional_test()
    print("All R‑Tree unit tests passed.")


if __name__ == "__main__":
    _run_all_tests()
