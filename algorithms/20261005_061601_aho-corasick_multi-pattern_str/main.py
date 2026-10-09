from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class _Node:
    children: Dict[str, _Node] = field(default_factory=dict)
    fail: Optional[_Node] = None
    outputs: List[str] = field(default_factory=list)


class AhoCorasick:
    """Aho‑Corasick multi‑pattern string searching automaton."""

    def __init__(self) -> None:
        self._root: _Node = _Node()
        self._built: bool = False

    def add(self, word: str) -> None:
        """Insert a pattern into the trie."""
        if not word:
            raise ValueError("Empty pattern is not allowed")
        node = self._root
        for ch in word:
            node = node.children.setdefault(ch, _Node())
        node.outputs.append(word)
        self._built = False

    def build(self) -> None:
        """Construct failure links and propagate output lists."""
        q: deque[_Node] = deque()
        for child in self._root.children.values():
            child.fail = self._root
            q.append(child)

        while q:
            current = q.popleft()
            for ch, child in current.children.items():
                q.append(child)
                fail_node = current.fail
                while fail_node is not None and ch not in fail_node.children:
                    fail_node = fail_node.fail
                child.fail = fail_node.children[ch] if fail_node and ch in fail_node.children else self._root
                child.outputs += child.fail.outputs
        self._built = True

    def search(self, text: str) -> List[Tuple[int, str]]:
        """
        Scan `text` and return a list of (start_index, pattern) matches.
        Overlapping matches are all reported.
        """
        if not self._built:
            self.build()
        node = self._root
        results: List[Tuple[int, str]] = []
        for i, ch in enumerate(text):
            while node is not None and ch not in node.children:
                node = node.fail
            if node is None:
                node = self._root
                continue
            node = node.children[ch]
            for pat in node.outputs:
                results.append((i - len(pat) + 1, pat))
        return results


def _run_tests() -> None:
    # Basic functionality test
    ac = AhoCorasick()
    patterns = ["he", "she", "his", "hers"]
    for p in patterns:
        ac.add(p)
    text = "ushers"
    matches = ac.search(text)
    expected = [(1, "she"), (2, "he"), (3, "hers")]
    assert sorted(matches) == sorted(expected), f"Basic test failed: {matches}"

    # Overlapping patterns test
    ac = AhoCorasick()
    for p in ["a", "ab", "bab", "bc", "bca", "c", "caa"]:
        ac.add(p)
    text = "abccab"
    matches = ac.search(text)
    expected = [
        (0, "a"), (0, "ab"),
        (1, "b"), (1, "bc"),
        (2, "c"), (2, "c"),
        (3, "c"), (3, "c"),
        (4, "a"), (4, "ab"),
        (5, "b")
    ]
    assert sorted(matches) == sorted(expected), f"Overlap test failed: {matches}"

    # No match test
    ac = AhoCorasick()
    ac.add("xyz")
    assert ac.search("abc") == [], "No‑match test failed"

    # Large random test (deterministic)
    ac = AhoCorasick()
    words = ["test", "testing", "ing", "est"]
    for w in words:
        ac.add(w)
    txt = "this is a testing test"
    res = ac.search(txt)
    exp = [
        (10, "test"), (10, "est"),
        (10, "testing"), (10, "ing"),
        (18, "test"), (18, "est")
    ]
    assert sorted(res) == sorted(exp), f"Large test failed: {res}"

    # Empty pattern rejection
    ac = AhoCorasick()
    try:
        ac.add("")
        assert False, "Empty pattern should raise"
    except ValueError:
        pass

    print("All tests passed.")


if __name__ == "__main__":
    _run_tests()
