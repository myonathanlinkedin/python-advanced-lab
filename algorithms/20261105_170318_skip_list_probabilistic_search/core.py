import random
from typing import Any, List, Optional, Tuple, Iterator


class SkipListNode:
    __slots__ = ("key", "value", "forward")

    def __init__(self, key: Any, value: Any, level: int) -> None:
        self.key = key
        self.value = value
        # forward[i] points to the next node at level i
        self.forward: List[Optional["SkipListNode"]] = [None] * (level + 1)


class SkipList:
    """
    Probabilistic Skip List supporting O(log n) expected search and insertion.
    Deterministic behaviour can be achieved by providing a seed.
    """

    def __init__(self, max_level: int = 16, p: float = 0.5, seed: Optional[int] = None) -> None:
        if not (0 < p < 1):
            raise ValueError("Probability p must be in (0,1).")
        if max_level < 1:
            raise ValueError("max_level must be >= 1.")
        self.max_level = max_level
        self.p = p
        self._rand = random.Random(seed)
        self.head = SkipListNode(None, None, max_level)
        self.level = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        return self.search(key) is not None

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        node = self.head.forward[0]
        while node:
            yield (node.key, node.value)
            node = node.forward[0]

    def _random_level(self) -> int:
        lvl = 0
        while lvl < self.max_level and self._rand.random() < self.p:
            lvl += 1
        return lvl

    def search(self, key: Any) -> Optional[Any]:
        """
        Returns the value associated with `key` or None if not present.
        """
        current = self.head
        for i in reversed(range(self.level + 1)):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        # Move to possible target node at level 0
        current = current.forward[0]
        if current and current.key == key:
            return current.value
        return None

    def insert(self, key: Any, value: Any) -> None:
        """
        Inserts `key` with `value`. If `key` already exists, its value is updated.
        """
        update: List[Optional[SkipListNode]] = [None] * (self.max_level + 1)
        current = self.head

        # Find positions that need to be updated
        for i in reversed(range(self.level + 1)):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        # Check if key already exists
        candidate = current.forward[0]
        if candidate and candidate.key == key:
            candidate.value = value
            return

        node_level = self._random_level()
        if node_level > self.level:
            for i in range(self.level + 1, node_level + 1):
                update[i] = self.head
            self.level = node_level

        new_node = SkipListNode(key, value, node_level)
        for i in range(node_level + 1):
            new_node.forward[i] = update[i].forward[i] if update[i] else None
            if update[i]:
                update[i].forward[i] = new_node
        self._size += 1

    def delete(self, key: Any) -> bool:
        """
        Removes `key` from the skip list.
        Returns True if the key was present, False otherwise.
        """
        update: List[Optional[SkipListNode]] = [None] * (self.max_level + 1)
        current = self.head

        for i in reversed(range(self.level + 1)):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        target = current.forward[0]
        if not target or target.key != key:
            return False

        for i in range(self.level + 1):
            if update[i].forward[i] != target:
                continue
            update[i].forward[i] = target.forward[i]

        # Adjust the current level of the list
        while self.level > 0 and self.head.forward[self.level] is None:
            self.level -= 1

        self._size -= 1
        return True

    def dump_levels(self) -> List[List[Any]]:
        """
        Utility for debugging: returns a list of levels with node keys.
        Level 0 is the bottom level.
        """
        result: List[List[Any]] = []
        for i in range(self.level + 1):
            level_nodes: List[Any] = []
            node = self.head.forward[i]
            while node:
                level_nodes.append(node.key)
                node = node.forward[i]
            result.append(level_nodes)
        return result
