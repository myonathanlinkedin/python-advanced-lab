"""
core.py
---------

Implementation of the Floyd‑Warshall algorithm for all‑pairs shortest paths
in a directed weighted graph, together with a utility to compute the
round‑trip distance matrix (i → j → i).

The module is deliberately pure: it uses only the Python standard library,
contains full type hints, and raises clear exceptions on invalid input
(e.g., negative‑weight cycles).
"""

from __future__ import annotations

from typing import List, Tuple

INF: float = float("inf")


def floyd_warshall(
    n: int, edges: List[Tuple[int, int, float]]
) -> List[List[float]]:
    """
    Compute the all‑pairs shortest‑path distance matrix for a directed graph.

    Parameters
    ----------
    n : int
        Number of vertices, assumed to be labelled 0 … n‑1.
    edges : List[Tuple[int, int, float]]
        Edge list where each tuple is (u, v, w) meaning an edge u → v with weight w.
        Multiple edges between the same ordered pair are allowed; the smallest
        weight is retained.

    Returns
    -------
    dist : List[List[float]]
        A square matrix ``dist[i][j]`` containing the length of the shortest
        directed path from *i* to *j*.  ``INF`` denotes that *j* is unreachable
        from *i*.

    Raises
    ------
    ValueError
        If a negative‑weight cycle is detected.
    """
    # Initialise distance matrix
    dist: List[List[float]] = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0

    # Insert edges, keeping the minimum weight for parallel edges
    for u, v, w in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise IndexError(f"Edge ({u}, {v}) contains invalid vertex index.")
        if w < dist[u][v]:
            dist[u][v] = w

    # Floyd‑Warshall triple loop
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            for j in range(n):
                dkj = dk[j]
                if dkj == INF:
                    continue
                new_dist = dik + dkj
                if new_dist < di[j]:
                    di[j] = new_dist

    # Detect negative cycles
    for i in range(n):
        if dist[i][i] < 0.0:
            raise ValueError("Graph contains a negative‑weight cycle.")

    return dist


def roundtrip_matrix(dist: List[List[float]]) -> List[List[float]]:
    """
    Given an all‑pairs shortest‑path matrix, compute the round‑trip distance
    matrix R where R[i][j] = dist[i][j] + dist[j][i] if both legs are finite,
    otherwise INF.

    Parameters
    ----------
    dist : List[List[float]]
        Square matrix produced by :func:`floyd_warshall`.

    Returns
    -------
    List[List[float]]
        Square matrix of round‑trip distances.
    """
    n = len(dist)
    if any(len(row) != n for row in dist):
        raise ValueError("Distance matrix must be square.")

    roundtrip: List[List[float]] = [[INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dij = dist[i][j]
            dji = dist[j][i]
            if dij != INF and dji != INF:
                roundtrip[i][j] = dij + dji
    return roundtrip


__all__ = ["INF", "floyd_warshall", "roundtrip_matrix"]
