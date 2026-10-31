from __future__ import annotations

from typing import List

from custom_types import Node


class SuffixAutomaton:
    """
    Linear‑time construction of a suffix automaton (SAM) for a given string.
    Provides O(|p|) substring existence queries and occurrence counting.

    The implementation follows the classic online construction algorithm
    (also known as the "extend" method) with explicit occurrence propagation.
    """

    def __init__(self, s: str = "") -> None:
        """
        Initialise an empty automaton and optionally build it from `s`.

        Parameters
        ----------
        s: str
            The source string. If empty, an empty automaton (only the root) is created.
        """
        # state 0 is the root
        self.states: List[Node] = [Node()]
        self.last: int = 0  # index of the state representing the whole string processed so far
        if s:
            self.build(s)

    # --------------------------------------------------------------------- #
    # Core construction primitives
    # --------------------------------------------------------------------- #
    def _add_state(self, length: int) -> int:
        """Create a new state with given `length` and return its index."""
        self.states.append(Node(length=length))
        return len(self.states) - 1

    def extend(self, ch: str) -> None:
        """
        Extend the automaton with a new character `ch`.

        This method updates the transition graph, creates clone states when
        necessary, and maintains suffix links. It also increments the occurrence
        counter of the newly created terminal state.
        """
        p = self.last
        cur = self._add_state(self.states[p].length + 1)
        self.states[cur].occ = 1  # each newly added state corresponds to a new end position

        # propagate transition for `ch` from existing states
        while p is not None and ch not in self.states[p].next:
            self.states[p].next[ch] = cur
            p = self.states[p].link

        if p is None:
            # reached root without a transition: link to root
            self.states[cur].link = 0
        else:
            q = self.states[p].next[ch]
            if self.states[p].length + 1 == self.states[q].length:
                # direct link is sufficient
                self.states[cur].link = q
            else:
                # need to clone state q
                clone = self._add_state(self.states[p].length + 1)
                self.states[clone].next = self.states[q].next.copy()
                self.states[clone].link = self.states[q].link
                # clone does not represent a new end position
                self.states[clone].occ = 0

                # redirect transitions pointing to q to point to clone
                while p is not None and self.states[p].next.get(ch) == q:
                    self.states[p].next[ch] = clone
                    p = self.states[p].link

                self.states[q].link = self.states[cur].link = clone

        self.last = cur

    def build(self, s: str) -> None:
        """Build the automaton for the entire string `s`."""
        for ch in s:
            self.extend(ch)
        self._propagate_occurrences()

    # --------------------------------------------------------------------- #
    # Post‑processing
    # --------------------------------------------------------------------- #
    def _propagate_occurrences(self) -> None:
        """
        After the whole string is inserted, propagate occurrence counts from longer
        states to their suffix links. This yields the exact number of end‑pos
        occurrences for each state.
        """
        # Sort states by decreasing length (topological order)
        order = sorted(range(len(self.states)), key=lambda i: self.states[i].length, reverse=True)
        for v in order:
            link = self.states[v].link
            if link is not None:
                self.states[link].occ += self.states[v].occ

    # --------------------------------------------------------------------- #
    # Query interface
    # --------------------------------------------------------------------- #
    def contains(self, substring: str) -> bool:
        """
        Determine whether `substring` occurs in the original text.

        Returns
        -------
        bool
            True if `substring` is a factor of the indexed string, False otherwise.
        """
        if substring == "":
            return True  # empty string is a substring of any string
        v = 0
        for ch in substring:
            if ch not in self.states[v].next:
                return False
            v = self.states[v].next[ch]
        return True

    def count_occurrences(self, substring: str) -> int:
        """
        Return the number of (possibly overlapping) occurrences of `substring`
        in the indexed text.

        Returns
        -------
        int
            Zero if `substring` does not occur.
        """
        if substring == "":
            # By definition, the empty string occurs len(text)+1 times.
            # The root state aggregates occurrences of all suffixes.
            return len(self.states) - 1  # number of non‑root states = number of positions
        v = 0
        for ch in substring:
            if ch not in self.states[v].next:
                return 0
            v = self.states[v].next[ch]
        return self.states[v].occ

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        """Return the number of states (including the root)."""
        return len(self.states)
