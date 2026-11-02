import threading
import random
import unittest
from typing import Set, List

from core import OptimisticLinkedList


class TestOptimisticLinkedList(unittest.TestCase):
    def setUp(self) -> None:
        self.lst = OptimisticLinkedList()

    def test_sequential_operations(self) -> None:
        # Insert a range of values
        for i in range(10):
            self.assertTrue(self.lst.insert(i))
        # Duplicate insertions must fail
        for i in range(10):
            self.assertFalse(self.lst.insert(i))
        # Contains checks
        for i in range(10):
            self.assertTrue(self.lst.contains(i))
        self.assertFalse(self.lst.contains(100))
        # Remove some values
        for i in range(0, 10, 2):
            self.assertTrue(self.lst.remove(i))
        for i in range(0, 10, 2):
            self.assertFalse(self.lst.contains(i))
        # Remaining values
        expected = [i for i in range(1, 10, 2)]
        self.assertEqual(self.lst.to_list(), expected)

    def test_concurrent_mixed_operations(self) -> None:
        # Shared set to track expected final contents
        expected: Set[int] = set()
        lock = threading.Lock()
        ops_per_thread = 1000
        thread_count = 8

        def worker(thread_id: int) -> None:
            local_rand = random.Random(thread_id)  # deterministic per thread
            for _ in range(ops_per_thread):
                op = local_rand.choice(["insert", "remove", "contains"])
                val = local_rand.randint(0, 50)
                if op == "insert":
                    inserted = self.lst.insert(val)
                    with lock:
                        if inserted:
                            expected.add(val)
                elif op == "remove":
                    removed = self.lst.remove(val)
                    with lock:
                        if removed:
                            expected.discard(val)
                else:  # contains
                    _ = self.lst.contains(val)  # result not used

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(thread_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify final list matches expected set
        final = set(self.lst.to_list())
        self.assertEqual(final, expected)

    def test_contains_optimistic_correctness(self) -> None:
        # Populate list
        for i in range(20):
            self.lst.insert(i)

        # Run many concurrent contains checks while other threads modify the list
        stop_flag = threading.Event()
        read_errors: List[str] = []

        def reader() -> None:
            while not stop_flag.is_set():
                for v in range(20):
                    present = self.lst.contains(v)
                    # The list may be concurrently modified, but contains must never raise
                    if not isinstance(present, bool):
                        read_errors.append(f"Non‑bool result for {v}")

        def writer() -> None:
            for _ in range(200):
                val = random.randint(0, 30)
                if random.random() < 0.5:
                    self.lst.insert(val)
                else:
                    self.lst.remove(val)

        reader_threads = [threading.Thread(target=reader) for _ in range(4)]
        writer_thread = threading.Thread(target=writer)

        for t in reader_threads:
            t.start()
        writer_thread.start()
        writer_thread.join()
        stop_flag.set()
        for t in reader_threads:
            t.join()

        self.assertEqual(len(read_errors), 0, f"Read errors: {read_errors}")

    def test_to_list_consistency(self) -> None:
        # Insert values concurrently and then snapshot
        values = list(range(30))
        random.shuffle(values)

        def inserter(vals: List[int]) -> None:
            for v in vals:
                self.lst.insert(v)

        threads = [
            threading.Thread(target=inserter, args=(values[i::3],))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lst = self.lst.to_list()
        self.assertEqual(lst, sorted(set(values)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
