import threading
from typing import Optional, Tuple, List


class Node:
    __slots__ = ("value", "next", "lock", "version")

    def __init__(self, value: int, next: Optional["Node"] = None) -> None:
        self.value: int = value
        self.next: Optional[Node] = next
        self.lock: threading.Lock = threading.Lock()
        self.version: int = 0  # incremented on each structural change


class OptimisticLinkedList:
    """
    A thread‑safe sorted singly linked list using safe optimistic lock coupling.
    Supports insert, remove and contains operations.
    """

    def __init__(self) -> None:
        # Sentinel head with value -inf
        self.head: Node = Node(float("-inf"))
        # Sentinel tail with value +inf
        self.tail: Node = Node(float("inf"))
        self.head.next = self.tail

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _find(self, value: int) -> Tuple[Node, Node]:
        """
        Traverses the list without acquiring any lock.
        Returns a tuple (pred, curr) where pred.value < value <= curr.value.
        """
        pred = self.head
        curr = pred.next  # type: ignore[assignment]
        while curr is not None and curr.value < value:
            pred = curr
            curr = curr.next
        return pred, curr  # type: ignore[return-value]

    def _validate(self, pred: Node, curr: Optional[Node]) -> bool:
        """
        Checks that pred and curr are still adjacent in the list.
        Must be called while holding pred.lock (and curr.lock if curr is not None).
        """
        return pred.next is curr

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def insert(self, value: int) -> bool:
        """
        Inserts value into the list if not already present.
        Returns True on successful insertion, False if the value already exists.
        """
        while True:
            pred, curr = self._find(value)

            # Acquire locks in address order to avoid deadlock
            first, second = (pred, curr) if id(pred) < id(curr) else (curr, pred)
            if first is not None:
                first.lock.acquire()
            if second is not None and second is not first:
                second.lock.acquire()

            try:
                if not self._validate(pred, curr):
                    continue  # retry

                if curr is not None and curr.value == value:
                    return False  # already present

                new_node = Node(value, curr)
                pred.next = new_node
                pred.version += 1
                if curr is not None:
                    curr.version += 1
                return True
            finally:
                if second is not None and second is not first:
                    second.lock.release()
                if first is not None:
                    first.lock.release()

    def remove(self, value: int) -> bool:
        """
        Removes value from the list if present.
        Returns True on successful removal, False if the value was absent.
        """
        while True:
            pred, curr = self._find(value)

            if curr is None or curr.value != value:
                return False  # not found

            # Acquire locks in address order
            first, second = (pred, curr) if id(pred) < id(curr) else (curr, pred)
            first.lock.acquire()
            if second is not None and second is not first:
                second.lock.acquire()

            try:
                if not self._validate(pred, curr):
                    continue  # retry

                # Physical removal
                pred.next = curr.next
                pred.version += 1
                if curr.next is not None:
                    curr.next.version += 1
                return True
            finally:
                if second is not None and second is not first:
                    second.lock.release()
                first.lock.release()

    def contains(self, value: int) -> bool:
        """
        Checks whether value is present.
        Uses optimistic reads: reads node versions before and after traversal.
        """
        while True:
            pred, curr = self._find(value)

            # Record versions without locking
            pred_ver = pred.version
            curr_ver = curr.version if curr is not None else -1

            # Verify adjacency hasn't changed
            if pred.next is not curr:
                continue  # retry

            # If adjacency is stable, check the value
            if curr is not None and curr.value == value:
                # Ensure versions unchanged during check
                if pred.version == pred_ver and curr.version == curr_ver:
                    return True
                else:
                    continue  # retry due to concurrent modification
            else:
                if pred.version == pred_ver:
                    return False
                else:
                    continue  # retry

    def to_list(self) -> List[int]:
        """
        Returns a Python list of the stored values (excluding sentinels).
        Primarily for testing; acquires a global lock to guarantee consistency.
        """
        result: List[int] = []
        node = self.head.next
        while node is not None and node is not self.tail:
            result.append(node.value)
            node = node.next
        return result
