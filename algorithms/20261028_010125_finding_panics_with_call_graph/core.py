"""
core.py
---------

Implementation of a simple Call Graph Analyzer that determines which functions
may cause a panic (directly or indirectly) based on a directed call graph.

The algorithm works in O(N + E) time where N is the number of functions and
E is the number of call edges.

Public API
~~~~~~~~~~
- CallGraph: main class representing the call graph.
    - add_function(name: str, panics: bool = False) -> None
    - add_call(caller: str, callee: str) -> None
    - set_panic(name: str, panics: bool = True) -> None
    - find_panic_functions() -> set[str]
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Set, Iterable


class CallGraph:
    """
    Represents a directed call graph of functions.

    Attributes
    ----------
    _adj : Dict[str, Set[str]]
        Outgoing adjacency list: caller -> set of callees.
    _rev_adj : Dict[str, Set[str]]
        Reverse adjacency list: callee -> set of callers.
    _panics : Dict[str, bool]
        Direct panic flag for each function.
    """

    def __init__(self) -> None:
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._rev_adj: Dict[str, Set[str]] = defaultdict(set)
        self._panics: Dict[str, bool] = {}

    # --------------------------------------------------------------------- #
    # Mutating API
    # --------------------------------------------------------------------- #
    def add_function(self, name: str, panics: bool = False) -> None:
        """
        Register a function in the graph.

        Parameters
        ----------
        name : str
            Identifier of the function.
        panics : bool, optional
            Whether the function can panic directly. Default is False.
        """
        if name not in self._panics:
            self._panics[name] = panics
        else:
            # Preserve existing direct panic flag unless explicitly overridden.
            self._panics[name] = self._panics[name] or panics
        # Ensure adjacency entries exist.
        self._adj.setdefault(name, set())
        self._rev_adj.setdefault(name, set())

    def add_call(self, caller: str, callee: str) -> None:
        """
        Add a directed edge representing a call from `caller` to `callee`.

        If either function is unknown, it is implicitly added with
        `panics=False`.

        Parameters
        ----------
        caller : str
            Name of the calling function.
        callee : str
            Name of the called function.
        """
        self.add_function(caller)
        self.add_function(callee)
        if callee not in self._adj[caller]:
            self._adj[caller].add(callee)
            self._rev_adj[callee].add(caller)

    def set_panic(self, name: str, panics: bool = True) -> None:
        """
        Explicitly set or clear the direct panic flag for a function.

        Parameters
        ----------
        name : str
            Function identifier.
        panics : bool, optional
            True to mark as panic‑capable, False to clear. Default True.
        """
        self.add_function(name)  # ensures entry exists
        self._panics[name] = panics

    # --------------------------------------------------------------------- #
    # Query API
    # --------------------------------------------------------------------- #
    def find_panic_functions(self) -> Set[str]:
        """
        Compute the set of functions that may panic, either directly or via
        transitive calls.

        Returns
        -------
        Set[str]
            All function names that can cause a panic when invoked.
        """
        # Direct panic functions form the initial frontier.
        panic_set: Set[str] = {fn for fn, flag in self._panics.items() if flag}
        if not panic_set:
            return set()

        # Perform reverse BFS from panic_set using the reverse adjacency.
        visited: Set[str] = set(panic_set)
        queue: deque[str] = deque(panic_set)

        while queue:
            cur = queue.popleft()
            for caller in self._rev_adj.get(cur, ()):
                if caller not in visited:
                    visited.add(caller)
                    queue.append(caller)

        return visited

    # --------------------------------------------------------------------- #
    # Utility methods (mostly for testing / introspection)
    # --------------------------------------------------------------------- #
    def functions(self) -> Set[str]:
        """Return the set of all known function names."""
        return set(self._panics.keys())

    def calls(self) -> Set[tuple[str, str]]:
        """Return the set of all call edges as (caller, callee) tuples."""
        return {(caller, callee) for caller, callees in self._adj.items() for callee in callees}

    def __repr__(self) -> str:
        return (
            f"CallGraph(num_functions={len(self._panics)}, "
            f"num_calls={sum(len(v) for v in self._adj.values())})"
        )
