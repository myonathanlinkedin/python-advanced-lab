import random
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional

MAX_LEVEL: int = 16
P: float = 0.5

@dataclass
class Node:
    key: Any
    value: Any
    forward: List[Optional["Node"]]

class SkipList:
    def __init__(self) -> None:
        self.head: Node = Node(None, None, [None] * MAX_LEVEL)
        self.level: int = 0
        self.size: int = 0

    def _random_level(self) -> int:
        lvl: int = 1
        while lvl < MAX_LEVEL and random.random() < P:
            lvl += 1
        return lvl

    def insert(self, key: Any, value: Any) -> None:
        update: List[Optional[Node]] = [None] * MAX_LEVEL
        current: Optional[Node] = self.head
        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if current and current.key == key:
            current.value = value
            return
        lvl = self._random_level()
        if lvl > self.level:
            for i in range(self.level, lvl):
                update[i] = self.head
            self.level = lvl
        new_node = Node(key, value, [None] * lvl)
        for i in range(lvl):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self.size += 1

    def search(self, key: Any) -> Optional[Any]:
        current: Optional[Node] = self.head
        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        if current and current.key == key:
            return current.value
        return None

    def delete(self, key: Any) -> bool:
        update: List[Optional[Node]] = [None] * MAX_LEVEL
        current: Optional[Node] = self.head
        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        target = current.forward[0]
        if not target or target.key != key:
            return False
        for i in range(self.level):
            if update[i].forward[i] != target:
                break
            update[i].forward[i] = target.forward[i]
        while self.level > 1 and self.head.forward[self.level - 1] is None:
            self.level -= 1
        self.size -= 1
        return True

    def __contains__(self, key: Any) -> bool:
        return self.search(key) is not None

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[Any]:
        current = self.head.forward[0]
        while current:
            yield current.key
            current = current.forward[0]

    def items(self) -> Iterator[tuple[Any, Any]]:
        current = self.head.forward[0]
        while current:
            yield (current.key, current.value)
            current = current.forward[0]

if __name__ == "__main__":
    # Demo
    sl = SkipList()
    for k in [3, 1, 4, 1, 5, 9, 2, 6, 5]:
        sl.insert(k, f"val{k}")
    print("Keys in order:", list(sl))
    print("Search 4:", sl.search(4))
    print("Search 7:", sl.search(7))
    print("Delete 5:", sl.delete(5))
    print("Keys after deletion:", list(sl))
    print("Size:", len(sl))

    # Unit tests
    test_list = SkipList()
    assert len(test_list) == 0
    test_list.insert(10, "ten")
    assert len(test_list) == 1
    assert test_list.search(10) == "ten"
    assert 10 in test_list
    test_list.insert(20, "twenty")
    test_list.insert(15, "fifteen")
    assert list(test_list) == [10, 15, 20]
    assert test_list.delete(15) is True
    assert list(test_list) == [10, 20]
    assert test_list.delete(30) is False
    test_list.insert(10, "TEN")
    assert test_list.search(10) == "TEN"
    print("All tests passed.")
