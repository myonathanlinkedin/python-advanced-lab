from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TrieNode:
    """A node in the Trie.

    Attributes
    ----------
    children: Mapping from character to child TrieNode.
    is_word: True if the node terminates a valid word.
    frequency: Cumulative frequency count for the word ending at this node.
    """
    children: Dict[str, TrieNode] = field(default_factory=dict)
    is_word: bool = False
    frequency: int = 0


class Trie:
    """Prefix tree supporting insertion, frequency updates and top‑k auto‑completion.

    The implementation is deliberately simple and deterministic:
    - Insertion and frequency updates are O(L) where L is the word length.
    - Autocompletion walks the sub‑trie of the given prefix, collects all words,
      then sorts them by frequency (descending) and lexicographically.
      This yields O(N_subtree log N_subtree) time where N_subtree is the number
      of words sharing the prefix.
    """

    def __init__(self) -> None:
        self._root = TrieNode()

    def insert(self, word: str, frequency: int = 1) -> None:
        """Insert *word* into the trie, adding *frequency* to its count.

        Parameters
        ----------
        word: non‑empty string consisting of any characters.
        frequency: positive integer to add to the word's count.

        Raises
        ------
        ValueError
            If *word* is empty or *frequency* is not positive.
        """
        if not word:
            raise ValueError("cannot insert empty word")
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        node = self._root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True
        node.frequency += frequency

    def increment(self, word: str, delta: int = 1) -> None:
        """Increase the frequency of *word* by *delta* (default 1).

        If the word does not exist, it is created with the given delta.
        """
        self.insert(word, delta)

    def _find_node(self, prefix: str) -> TrieNode | None:
        """Return the node representing *prefix* or None if not present."""
        node = self._root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def autocomplete(self, prefix: str, k: int = 10) -> List[Tuple[str, int]]:
        """Return up to *k* words that start with *prefix*, ordered by frequency.

        The result list contains tuples ``(word, frequency)``.  Ordering rules:
        1. Higher frequency first.
        2. For equal frequencies, lexicographically smaller word first.

        Parameters
        ----------
        prefix: string prefix; may be empty to request the global top‑k.
        k: maximum number of suggestions; must be non‑negative.

        Returns
        -------
        List[Tuple[str, int]]
            Sorted suggestions; possibly fewer than *k* if not enough matches.
        """
        if k < 0:
            raise ValueError("k must be non‑negative")
        start_node = self._root if prefix == "" else self._find_node(prefix)
        if start_node is None:
            return []

        results: List[Tuple[str, int]] = []

        def dfs(node: TrieNode, path: List[str]) -> None:
            if node.is_word:
                word = "".join(path)
                results.append((word, node.frequency))
            for ch, child in node.children.items():
                path.append(ch)
                dfs(child, path)
                path.pop()

        dfs(start_node, list(prefix))
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:k]

    def __contains__(self, word: str) -> bool:
        node = self._find_node(word)
        return bool(node and node.is_word)

    def __len__(self) -> int:
        """Return the number of distinct words stored in the trie."""
        count = 0

        def dfs(node: TrieNode) -> None:
            nonlocal count
            if node.is_word:
                count += 1
            for child in node.children.values():
                dfs(child)

        dfs(self._root)
        return count

    def get_frequency(self, word: str) -> int:
        """Return the stored frequency for *word*; 0 if the word is absent."""
        node = self._find_node(word)
        return node.frequency if node and node.is_word else 0
