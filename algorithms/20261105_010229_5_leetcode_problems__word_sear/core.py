from typing import List, Dict

class WordSearchSolver:
    """
    Solver for LeetCode 79 - Word Search.
    Provides a method `exist` that determines if a given word can be formed
    by sequentially adjacent cells (horizontally or vertically) in the board.
    Each cell may be used at most once per search path.
    """

    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board or not board[0] or not word:
            return False

        self.rows, self.cols = len(board), len(board[0])
        self.board = board
        self.word = word

        for r in range(self.rows):
            for c in range(self.cols):
                if board[r][c] == word[0] and self._dfs(r, c, 0):
                    return True
        return False

    def _dfs(self, r: int, c: int, idx: int) -> bool:
        """Depth‑first search with backtracking."""
        if idx == len(self.word):
            return True
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        if self.board[r][c] != self.word[idx]:
            return False

        # Mark the cell as visited by temporary alteration
        temp = self.board[r][c]
        self.board[r][c] = '#'

        # Explore four directions
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if self._dfs(r + dr, c + dc, idx + 1):
                self.board[r][c] = temp
                return True

        # Restore original value
        self.board[r][c] = temp
        return False


class TrieNode:
    __slots__ = ('children', 'is_end')

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False


class Trie:
    """
    Implementation of a prefix tree (Trie) supporting insert, search,
    and startsWith operations in O(L) time where L is the length of the key.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node.is_end if node else False

    def startsWith(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode | None:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node


def count_palindromic_substrings(s: str) -> int:
    """
    Returns the number of palindromic substrings in `s`.
    Uses the expand‑around‑center technique with O(n²) time and O(1) extra space.
    """
    n = len(s)
    count = 0

    for center in range(2 * n - 1):
        left = center // 2
        right = left + center % 2
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    return count
