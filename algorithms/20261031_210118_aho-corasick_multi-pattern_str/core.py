from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Iterable, Optional


class AhoCorasickAutomaton:
    """
    Production‑grade implementation of the Aho‑Corasick multi‑pattern string searching automaton.

    The automaton is built in two phases:
        1. Insert all patterns (add_pattern).
        2. Build failure links (build).

    After building, the `search` method scans a text in O(len(text)) time and yields
    (position, matched_pattern) tuples where `position` is the index of the *last*
    character of the match in the original text.
    """

    __slots__ = ("_goto", "_output", "_fail", "_built", "_patterns")

    def __init__(self) -> None:
        # Transition table: node -> {char: next_node}
        self._goto: List[Dict[str, int]] = [{}]
        # Output list: node -> list of pattern indices ending at this node
        self._output: List[List[int]] = [[]]
        # Failure function: node -> node
        self._fail: List[int] = [0]
        # Patterns stored for result reconstruction
        self._patterns: List[str] = []
        # Build flag
        self._built: bool = False

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #
    def add_pattern(self, pattern: str) -> None:
        """
        Insert a pattern into the trie. Must be called before `build`.
        Empty patterns are ignored.
        """
        if not pattern:
            return
        node = 0
        for ch in pattern:
            if ch not in self._goto[node]:
                self._goto.append({})
                self._output.append([])
                self._fail.append(0)
                self._goto[node][ch] = len(self._goto) - 1
            node = self._goto[node][ch]
        # Store pattern index in output list of the terminal node
        pattern_index = len(self._patterns)
        self._patterns.append(pattern)
        self._output[node].append(pattern_index)

    def build(self) -> None:
        """
        Construct failure links using BFS. Must be called after all patterns are added.
        """
        if self._built:
            return
        queue: deque[int] = deque()
        # Initialize depth‑1 nodes: fail to root (0)
        for ch, nxt in self._goto[0].items():
            self._fail[nxt] = 0
            queue.append(nxt)

        while queue:
            r = queue.popleft()
            for ch, u in self._goto[r].items():
                queue.append(u)
                # Follow failure links from parent
                state = self._fail[r]
                while state and ch not in self._goto[state]:
                    state = self._fail[state]
                self._fail[u] = self._goto[state].get(ch, 0)
                # Merge output
                self._output[u].extend(self._output[self._fail[u]])

        self._built = True

    # --------------------------------------------------------------------- #
    # Search
    # --------------------------------------------------------------------- #
    def search(self, text: str) -> Iterable[Tuple[int, str]]:
        """
        Scan `text` and yield (end_index, pattern) for each match.
        The automaton must be built before calling this method.
        """
        if not self._built:
            raise RuntimeError("Automaton must be built before searching.")
        state = 0
        for i, ch in enumerate(text):
            while state and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            for pat_idx in self._output[state]:
                yield (i, self._patterns[pat_idx])

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        """Number of stored patterns."""
        return len(self._patterns)

    def patterns(self) -> List[str]:
        """Return a copy of the pattern list."""
        return list(self._patterns)

    def reset(self) -> None:
        """Reset the automaton to an empty state."""
        self.__init__()
