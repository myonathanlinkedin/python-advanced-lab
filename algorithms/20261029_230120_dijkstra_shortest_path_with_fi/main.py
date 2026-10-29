from __future__ import annotations
from typing import Dict, List, Tuple
import math
from engine import dijkstra
import sys

def _assert_almost_equal(a: float, b: float, eps: float = 1e-9) -> None:
    assert abs(a - b) <= eps, f'{a} !~= {b}'

def test_simple_graph() -> None:
    # Graph:
    # 0 --1--> 1 --2--> 2
    # 0 --4--> 2
    graph: Dict[int, List[Tuple[int, float]]] = {
        0: [(1, 1.0), (2, 4.0)],
        1: [(2, 2.0)],
        2: []
    }
    dist, prev = dijkstra(graph, 0)
    _assert_almost_equal(dist[0], 0.0)
    _assert_almost_equal(dist[1], 1.0)
    _assert_almost_equal(dist[2], 3.0)
    assert prev[0] is None
    assert prev[1] == 0
    assert prev[2] == 1

def test_disconnected() -> None:
    graph = {
        0: [(1, 5.0)],
        1: [],
        2: [(3, 2.0)],
        3: []
    }
    dist, prev = dijkstra(graph, 0)
    _assert_almost_equal(dist[0], 0.0)
    _assert_almost_equal(dist[1], 5.0)
    assert math.isinf(dist[2])
    assert math.isinf(dist[3])
    assert prev[2] is None
    assert prev[3] is None

def test_single_node() -> None:
    graph = {0: []}
    dist, prev = dijkstra(graph, 0)
    _assert_almost_equal(dist[0], 0.0)
    assert prev[0] is None

def test_multiple_paths() -> None:
    # 0 to 3 via two paths:
    # 0-1-3 (1+1) = 2
    # 0-2-3 (0.5+0.5) = 1
    graph = {
        0: [(1, 1.0), (2, 0.5)],
        1: [(3, 1.0)],
        2: [(3, 0.5)],
        3: []
    }
    dist, prev = dijkstra(graph, 0)
    _assert_almost_equal(dist[3], 1.0)
    assert prev[3] == 2

def test_negative_weight_error() -> None:
    graph = {
        0: [(1, -1.0)],
        1: []
    }
    try:
        dijkstra(graph, 0)
    except ValueError as e:
        assert 'negative' in str(e).lower()
    else:
        assert False, 'Expected ValueError for negative weight'

def run_all_tests() -> None:
    test_simple_graph()
    test_disconnected()
    test_single_node()
    test_multiple_paths()
    test_negative_weight_error()
    print('All Dijkstra + Fibonacci Heap tests passed.')

if __name__ == '__main__':
    run_all_tests()
