from __future__ import annotations
import threading
import random
import time
from typing import List
from engine import OptimisticLockCouplingLinkedList

def sequential_test() -> None:
    lst = OptimisticLockCouplingLinkedList[int]()
    assert lst.to_list() == []

    # Insert a few elements
    assert lst.insert(10) is True
    assert lst.insert(5) is True
    assert lst.insert(20) is True
    assert lst.insert(10) is False  # duplicate

    # Verify order
    assert lst.to_list() == [5, 10, 20]

    # Contains checks
    assert lst.contains(5) is True
    assert lst.contains(15) is False

    # Delete operations
    assert lst.delete(5) is True
    assert lst.delete(5) is False  # already gone
    assert lst.delete(15) is False  # never existed

    assert lst.to_list() == [10, 20]

    # Clean up
    assert lst.delete(10) is True
    assert lst.delete(20) is True
    assert lst.to_list() == []

def concurrent_worker(lst: OptimisticLockCouplingLinkedList[int],
                      ops: List[tuple[str, int]],
                      results: List[bool],
                      idx: int) -> None:
    """
    Executes a sequence of operations on the shared list.
    Stores the boolean result of each operation in `results`.
    """
    local_res = []
    for op, val in ops:
        if op == 'i':
            local_res.append(lst.insert(val))
        elif op == 'd':
            local_res.append(lst.delete(val))
        elif op == 'c':
            local_res.append(lst.contains(val))
        else:
            raise ValueError(f'Unknown op {op}')
    results[idx] = all(local_res)  # simple aggregate for demo

def concurrent_test() -> None:
    random.seed(42)
    lst = OptimisticLockCouplingLinkedList[int]()

    # Pre‑populate list with even numbers 0..98
    for i in range(0, 100, 2):
        assert lst.insert(i) is True

    thread_count = 8
    ops_per_thread = 1000
    threads: List[threading.Thread] = []
    results = [False] * thread_count

    for t_idx in range(thread_count):
        ops = []
        for _ in range(ops_per_thread):
            op_type = random.choice(['i', 'd', 'c'])
            value = random.randint(0, 120)  # some values outside initial range
            ops.append((op_type, value))
        th = threading.Thread(target=concurrent_worker,
                              args=(lst, ops, results, t_idx))
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    # After all threads finish, verify list invariants:
    # 1. Sorted order
    # 2. No duplicates
    final = lst.to_list()
    assert final == sorted(final), "List is not sorted"
    assert len(final) == len(set(final)), "List contains duplicates"

    # 3. All present keys are indeed in the list according to contains()
    for key in final:
        assert lst.contains(key) is True

    # 4. Keys that were never inserted (e.g., 121) are absent
    assert lst.contains(121) is False

    # Simple sanity check that at least one thread succeeded in all its ops
    assert any(results), "All threads reported failure, unexpected"

if __name__ == '__main__':
    # Run deterministic unit tests
    sequential_test()
    concurrent_test()
    print('All tests passed.')
