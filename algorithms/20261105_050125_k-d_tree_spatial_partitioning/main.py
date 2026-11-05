import math
from core import KDTree, Point

def _assert_equal(a, b, msg=""):
    assert a == b, msg or f"{a!r} != {b!r}"

def _assert_close(a, b, tol=1e-9, msg=""):
    assert abs(a - b) <= tol, msg or f"{a!r} not close to {b!r}"

def test_empty_tree():
    tree = KDTree()
    _assert_equal(len(tree), 0)
    _assert_equal(tree.nearest_neighbor((0, 0)), None)

def test_single_point():
    tree = KDTree([(1.0, 2.0)])
    _assert_equal(len(tree), 1)
    _assert_equal(tree.nearest_neighbor((1.0, 2.0)), (1.0, 2.0))
    _assert_equal(tree.nearest_neighbor((0.0, 0.0)), (1.0, 2.0))

def test_multiple_points_2d():
    points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = KDTree(points)
    _assert_equal(len(tree), len(points))
    _assert_equal(tree.nearest_neighbor((9, 2)), (8, 1))
    _assert_equal(tree.nearest_neighbor((2, 2)), (2, 3))
    _assert_equal(tree.nearest_neighbor((5, 5)), (5, 4))

def test_multiple_points_3d():
    points = [
        (2, 3, 5),
        (5, 4, 1),
        (9, 6, 7),
        (4, 7, 2),
        (8, 1, 3),
        (7, 2, 9),
    ]
    tree = KDTree(points)
    _assert_equal(len(tree), len(points))
    _assert_equal(tree.nearest_neighbor((9, 2, 4)), (8, 1, 3))
    _assert_equal(tree.nearest_neighbor((2, 2, 2)), (2, 3, 5))
    _assert_equal(tree.nearest_neighbor((5, 5, 5)), (5, 4, 1))

def test_insert_and_search():
    tree = KDTree()
    tree.insert((1, 1))
    tree.insert((2, 2))
    tree.insert((3, 3))
    _assert_equal(len(tree), 3)
    _assert_equal(tree.nearest_neighbor((2.1, 2.1)), (2, 2))
    _assert_equal(tree.nearest_neighbor((0, 0)), (1, 1))

def test_dimension_mismatch():
    tree = KDTree([(1, 2)])
    try:
        tree.insert((1, 2, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for dimension mismatch")

def test_build_with_mixed_dimensions():
    try:
        KDTree([(1, 2), (3, 4, 5)])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mixed dimensions")

def test_contains():
    tree = KDTree([(1, 2), (3, 4)])
    _assert_equal((1, 2) in tree, True)
    _assert_equal((5, 6) in tree, False)

def test_iteration():
    points = [(1, 2), (3, 4), (5, 6)]
    tree = KDTree(points)
    collected = set(tree)
    _assert_equal(collected, set(points))

def test_nearest_neighbor_same_point():
    tree = KDTree([(1, 1), (2, 2), (3, 3)])
    _assert_equal(tree.nearest_neighbor((2, 2)), (2, 2))

def test_nearest_neighbor_multiple_same_distance():
    tree = KDTree([(0, 0), (0, 2), (2, 0)])
    # Query at (1, 1) equidistant to all three; any is acceptable
    result = tree.nearest_neighbor((1, 1))
    _assert_equal(result in [(0, 0), (0, 2), (2, 0)], True)

def test_large_number_of_points():
    points = [(float(i), float(i * 2)) for i in range(1000)]
    tree = KDTree(points)
    _assert_equal(len(tree), 1000)
    _assert_equal(tree.nearest_neighbor((500.5, 1001)), (500, 1000))

def run_all_tests():
    test_empty_tree()
    test_single_point()
    test_multiple_points_2d()
    test_multiple_points_3d()
    test_insert_and_search()
    test_dimension_mismatch()
    test_build_with_mixed_dimensions()
    test_contains()
    test_iteration()
    test_nearest_neighbor_same_point()
    test_nearest_neighbor_multiple_same_distance()
    test_large_number_of_points()
    print("All tests passed.")

def demo():
    print("Demo: KDTree nearest neighbor search")
    points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = KDTree(points)
    query = (9, 2)
    nearest = tree.nearest_neighbor(query)
    print(f"Points: {points}")
    print(f"Query point: {query}")
    print(f"Nearest neighbor: {nearest}")

if __name__ == "__main__":
    run_all_tests()
    demo()
