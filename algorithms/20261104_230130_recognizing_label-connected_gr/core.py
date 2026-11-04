from __future__ import annotations
from typing import Dict, Set, Iterable, Any


def is_label_connected(
    adjacency: Dict[Any, Set[Any]],
    labels: Dict[Any, Any],
) -> bool:
    """
    Determine whether a vertex‑labelled graph is label‑connected.

    A graph is label‑connected if for every label ℓ, the subgraph induced by
    vertices with label ℓ is connected.  The graph is represented by an
    adjacency dictionary mapping each vertex to the set of its neighbors.
    The labels dictionary maps each vertex to its label.

    Parameters
    ----------
    adjacency : Dict[Any, Set[Any]]
        Adjacency list of the graph.  Every vertex appearing in ``labels``
        must be a key in ``adjacency``.
    labels : Dict[Any, Any]
        Mapping from vertex to label.

    Returns
    -------
    bool
        ``True`` if the graph is label‑connected, ``False`` otherwise.

    Raises
    ------
    ValueError
        If the set of vertices in ``adjacency`` and ``labels`` differ.
    """
    if set(adjacency) != set(labels):
        raise ValueError("Vertices in adjacency and labels must match")

    # Group vertices by label
    label_to_vertices: Dict[Any, Set[Any]] = {}
    for v, lbl in labels.items():
        label_to_vertices.setdefault(lbl, set()).add(v)

    # For each label, check that its induced subgraph is connected
    for lbl, verts in label_to_vertices.items():
        if len(verts) <= 1:
            continue  # trivially connected

        start = next(iter(verts))
        visited: Set[Any] = set()
        stack: list[Any] = [start]

        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for w in adjacency[u]:
                if w in verts and w not in visited:
                    stack.append(w)

        if visited != verts:
            return False

    return True
