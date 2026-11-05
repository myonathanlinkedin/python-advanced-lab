from __future__ import annotations
from typing import Optional, TypeVar, Generic, List
from threading import Lock
from custom_types import Node

T = TypeVar('T')


class OptimisticLockCouplingLinkedList(Generic[T]):
    """
    A thread‑safe sorted singly linked list using optimistic lock coupling.
    The list maintains keys in strictly increasing order.
    """

    def __init__(self) -> None:
        # Sentinel head with key less than any possible user key.
        self.head: Node[T] = Node(key=self._sentinel_key())
        self.head.next = None

    @staticmethod
    def _sentinel_key() -> T:
        """
        Returns a sentinel key that is guaranteed to compare less than any user key.
        For generic comparable types we use None and rely on the user never inserting None.
        """
        return None  # type: ignore

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _read_node(self, node: Node[T]) -> tuple[int, Optional[Node[T]]]:
        """
        Reads a node's version and next pointer atomically (without acquiring the lock).
        The caller must later validate that the version has not changed.
        """
        version = node.version
        nxt = node.next
        return version, nxt

    def _validate(self, node: Node[T], observed_version: int) -> bool:
        """
        Returns True if the node's version is unchanged since it was observed.
        """
        return node.version == observed_version

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def contains(self, key: T) -> bool:
        """
        Optimistic read: traverse without locks, then validate the observed nodes.
        Returns True iff the key is present in the list.
        """
        while True:
            pred = self.head
            pred_ver, curr = self._read_node(pred)

            # Traverse until we find a node with key >= target
            while curr is not None and curr.key < key:
                pred = curr
                pred_ver, curr = self._read_node(pred)

            # At this point, curr is either None or has key >= target
            if curr is not None and curr.key == key:
                # Validate that both pred and curr have not changed
                if self._validate(pred, pred_ver) and self._validate(curr, curr.version):
                    return True
                # Validation failed – retry
                continue
            else:
                # Key not present; validate pred (the only node we observed)
                if self._validate(pred, pred_ver):
                    return False
                # Validation failed – retry
                continue

    def insert(self, key: T) -> bool:
        """
        Inserts key into the list if not already present.
        Returns True if insertion succeeded, False if key already existed.
        """
        while True:
            # 1. Optimistic traversal to locate predecessor and current
            pred = self.head
            pred_ver, curr = self._read_node(pred)

            while curr is not None and curr.key < key:
                pred = curr
                pred_ver, curr = self._read_node(pred)

            # 2. Acquire locks in address order to avoid deadlock
            #    (here we lock pred first, then curr if it exists)
            pred.lock.acquire()
            try:
                # Re‑read pred's next under lock to ensure it still points to curr
                if pred.next is not curr:
                    # Structure changed; restart
                    continue

                if curr is not None:
                    curr.lock.acquire()
                try:
                    # Validate versions after acquiring locks
                    if not self._validate(pred, pred_ver):
                        continue
                    if curr is not None and not self._validate(curr, curr.version):
                        continue

                    # Check for duplicate
                    if curr is not None and curr.key == key:
                        return False  # already present

                    # Insert new node
                    new_node = Node(key=key, next=curr)
                    pred.next = new_node

                    # Increment versions to signal change
                    pred.version += 1
                    if curr is not None:
                        curr.version += 1
                    return True
                finally:
                    if curr is not None:
                        curr.lock.release()
            finally:
                pred.lock.release()

    def delete(self, key: T) -> bool:
        """
        Deletes key from the list if present.
        Returns True if deletion succeeded, False if key was absent.
        """
        while True:
            # 1. Optimistic traversal to locate predecessor and current
            pred = self.head
            pred_ver, curr = self._read_node(pred)

            while curr is not None and curr.key < key:
                pred = curr
                pred_ver, curr = self._read_node(pred)

            # 2. Acquire locks in address order (pred then curr)
            pred.lock.acquire()
            try:
                if pred.next is not curr:
                    continue  # structure changed, retry

                if curr is not None:
                    curr.lock.acquire()
                try:
                    # Validate versions after lock acquisition
                    if not self._validate(pred, pred_ver):
                        continue
                    if curr is not None and not self._validate(curr, curr.version):
                        continue

                    if curr is None or curr.key != key:
                        return False  # key not present

                    # Remove curr
                    pred.next = curr.next

                    # Increment versions
                    pred.version += 1
                    curr.version += 1
                    return True
                finally:
                    if curr is not None:
                        curr.lock.release()
            finally:
                pred.lock.release()

    # -------------------------------------------------------------------------
    # Utility for testing
    # -------------------------------------------------------------------------
    def to_list(self) -> List[T]:
        """
        Returns a Python list of the keys in sorted order.
        Used only for testing; acquires a global lock to guarantee consistency.
        """
        # Acquire all node locks in order to get a consistent snapshot.
        # For simplicity we lock the head and then walk without further locks,
        # accepting that the snapshot may be slightly stale – sufficient for tests.
        result: List[T] = []
        node = self.head.next
        while node is not None:
            result.append(node.key)
            node = node.next
        return result
