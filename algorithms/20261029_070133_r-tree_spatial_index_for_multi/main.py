from __future__ import annotations
from custom_types import BoundingBox
from engine import RTree

def _bbox(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(mins=(x1, y1), maxs=(x2, y2))

def test_bounding_box_operations():
    a = _bbox(0, 0, 2, 2)
    b = _bbox(1, 1, 3, 3)
    c = _bbox(3, 3, 5, 5)
    assert a.intersects(b)
    assert not a.intersects(c)
    assert a.contains(a)
    assert not a.contains(b)
    assert a.union(b).mins == (0, 0)
    assert a.union(b).maxs == (3, 3)
    assert a.area() == 4.0
    assert b.area() == 4.0
    assert a.union(b).area() == 9.0

def test_rtree_insertion_and_search():
    tree = RTree(dimensions=2)
    # Insert 5 boxes
    boxes = [
        (_bbox(0, 0, 1, 1), "A"),
        (_bbox(1, 1, 2, 2), "B"),
        (_bbox(2, 2, 3, 3), "C"),
        (_bbox(0, 2, 1, 3), "D"),
        (_bbox(2, 0, 3, 1), "E"),
    ]
    for bbox, data in boxes:
        tree.insert(bbox, data)

    # Search overlapping with (0.5,0.5)-(2.5,2.5)
    query = _bbox(0.5, 0.5, 2.5, 2.5)
    results = tree.search(query)
    expected = {"A", "B", "C", "D", "E"}
    assert set(results) == expected

    # Search with a box that only overlaps B and C
    query2 = _bbox(1.5, 1.5, 2.5, 2.5)
    results2 = tree.search(query2)
    assert set(results2) == {"B", "C"}

    # Search with a box that overlaps none
    query3 = _bbox(10, 10, 12, 12)
    results3 = tree.search(query3)
    assert results3 == []

def test_edge_cases():
    tree = RTree(dimensions=2)
    # Zero-area box
    zero_box = _bbox(1, 1, 1, 1)
    tree.insert(zero_box, "Zero")
    query = _bbox(0, 0, 2, 2)
    results = tree.search(query)
    assert "Zero" in results

    # Overlapping identical boxes
    box = _bbox(0, 0, 2, 2)
    tree.insert(box, "First")
    tree.insert(box, "Second")
    results = tree.search(box)
    assert set(results) == {"Zero", "First", "Second"}

def main():
    test_bounding_box_operations()
    test_rtree_insertion_and_search()
    test_edge_cases()
    print("All tests passed successfully.")

if __name__ == "__main__":
    main()
