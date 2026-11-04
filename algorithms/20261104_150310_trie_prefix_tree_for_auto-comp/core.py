from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import heapq


class TrieNode:
    __slots__ = ("children", "is_end", "frequency")

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.frequency: int = 0  # cumulative frequency for the word ending here


class Trie:
    """
    Prefix tree (Trie) supporting insertion of words with frequencies and
    auto‑completion queries that return the *k* most frequent completions.
    """

    def __init__(self) -> None:
        self._root = TrieNode()

    def insert(self, word: str, freq: int = 1) -> None:
        """
        Insert *word* into the trie, adding *freq* to its stored frequency.
        If the word already exists, its frequency is incremented.
        """
        if not isinstance(word, str):
            raise TypeError("word must be a string")
        if not isinstance(freq, int) or freq <= 0:
            raise ValueError("freq must be a positive integer")

        node = self._root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
        node.frequency += freq

    def get_frequency(self, word: str) -> int:
        """
        Return the stored frequency of *word*; 0 if the word is absent.
        """
        node = self._find_node(word)
        return node.frequency if node and node.is_end else 0

    def autocomplete(self, prefix: str, k: int = 10) -> List[Tuple[str, int]]:
        """
        Return up to *k* (word, frequency) pairs that start with *prefix*,
        ordered by descending frequency then lexicographically.
        """
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        start_node = self._find_node(prefix)
        if start_node is None:
            return []

        # Gather all completions under start_node
        completions: List[Tuple[int, str]] = []  # (frequency, word)
        stack: List[Tuple[TrieNode, str]] = [(start_node, prefix)]

        while stack:
            node, cur_word = stack.pop()
            if node.is_end:
                completions.append((node.frequency, cur_word))
            for ch, child in node.children.items():
                stack.append((child, cur_word + ch))

        # Use heapq.nlargest for O(N log k) selection
        top_k = heapq.nlargest(
            k,
            completions,
            key=lambda item: (item[0], -ord(item[1][0]) if item[1] else 0),
        )
        # Sort final list to guarantee deterministic order (freq desc, word asc)
        top_k.sort(key=lambda item: (-item[0], item[1]))
        return [(word, freq) for freq, word in [(f, w) for f, w in top_k]]

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """
        Return the node representing *prefix* or None if the path does not exist.
        """
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node
