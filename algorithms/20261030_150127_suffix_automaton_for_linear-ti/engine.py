from __future__ import annotations
from typing import List, Iterable

from .types import State


class SuffixAutomaton:
    """
    Linear‑time construction of a suffix automaton (SAM) for a given string.
    Provides O(|p|) substring existence queries and O(|S|) distinct‑substring count.
    """

    def __init__(self, s: str = "") -> None:
        self.states: List[State] = []
        self._last: int = 0                     # Index of the state representing the whole string processed so far
        self._size: int = 0                     # Number of states currently allocated
        self._init_automaton()
        for ch in s:
            self._extend(ch)

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #
    def _init_automaton(self) -> None:
        """Create the initial root state."""
        root = State(length=0, link=None, first_pos=-1)
        self.states.append(root)
        self._last = 0
        self._size = 1

    def _extend(self, c: str) -> None:
        """
        Add character `c` to the automaton, updating the structure.
        Complexity: amortized O(1).
        """
        p = self._last
        cur = self._size
        self.states.append(State(length=self.states[p].length + 1,
                                 first_pos=self.states[p].length,
                                 is_clone=False))
        self._size += 1

        # Propagate transition for `c` from states that do not have it yet
        while p is not None and c not in self.states[p].next:
            self.states[p].next[c] = cur
            p = self.states[p].link

        if p is None:
            # No state with transition `c`; link to root
            self.states[cur].link = 0
        else:
            q = self.states[p].next[c]
            if self.states[p].length + 1 == self.states[q].length:
                # Directly link
                self.states[cur].link = q
            else:
                # Clone state `q`
                clone = self._size
                cloned_state = State(
                    length=self.states[p].length + 1,
                    link=self.states[q].link,
                    next=self.states[q].next.copy(),
                    first_pos=self.states[q].first_pos,
                    is_clone=True,
                )
                self.states.append(cloned_state)
                self._size += 1

                # Redirect transitions pointing to `q` to the clone
                while p is not None and self.states[p].next.get(c) == q:
                    self.states[p].next[c] = clone
                    p = self.states[p].link

                self.states[q].link = clone
                self.states[cur].link = clone

        self._last = cur

    # --------------------------------------------------------------------- #
    # Public query interface
    # --------------------------------------------------------------------- #
    def contains(self, substring: str) -> bool:
        """
        Return True iff `substring` occurs in the original string.
        Walks the automaton following transitions; O(|substring|).
        """
        v = 0
        for ch in substring:
            if ch not in self.states[v].next:
                return False
            v = self.states[v].next[ch]
        return True

    def count_distinct_substrings(self) -> int:
        """
        Count the number of distinct substrings of the original string.
        Formula: sum_{v != root} (len(v) - len(link(v))).
        Runs in O(|states|) = O(|S|).
        """
        total = 0
        for idx, st in enumerate(self.states):
            if idx == 0:
                continue  # root contributes nothing
            link_len = self.states[st.link].length if st.link is not None else 0
            total += st.length - link_len
        return total

    # --------------------------------------------------------------------- #
    # Utility / debugging
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        """Number of states in the automaton."""
        return self._size

    def iter_states(self) -> Iterable[State]:
        """Iterate over all states (including root)."""
        return iter(self.states)
