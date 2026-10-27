import unittest
from typing import List

from core import Process, ProcessState, Scheduler, MemoryManager


class TestScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = Scheduler(quantum=3)

    def test_simple_execution(self) -> None:
        p1 = Process(pid=1, total_time=5)
        p2 = Process(pid=2, total_time=2)
        self.scheduler.add_process(p1)
        self.scheduler.add_process(p2)

        # Expected order: p1 runs 3 ticks, preempted, p2 runs 2 ticks, finishes,
        # then p1 runs remaining 2 ticks.
        timeline: List[int] = []
        while not self.scheduler.is_idle():
            proc = self.scheduler.tick()
            if proc:
                timeline.append(proc.pid)

        self.assertEqual(timeline, [1, 1, 1, 2, 2, 1, 1])
        self.assertEqual(p1.state, ProcessState.TERMINATED)
        self.assertEqual(p2.state, ProcessState.TERMINATED)

    def test_preempt_and_termination(self) -> None:
        p = Process(pid=42, total_time=1)
        self.scheduler.add_process(p)
        proc = self.scheduler.tick()
        self.assertIsNotNone(proc)
        self.assertEqual(proc.pid, 42)
        self.assertEqual(p.state, ProcessState.TERMINATED)
        self.assertTrue(self.scheduler.is_idle())

    def test_invalid_add(self) -> None:
        p = Process(pid=7, total_time=3)
        p.state = ProcessState.RUNNING
        with self.assertRaises(ValueError):
            self.scheduler.add_process(p)

    def test_quantum_validation(self) -> None:
        with self.assertRaises(ValueError):
            Scheduler(quantum=0)


class TestMemoryManager(unittest.TestCase):
    def setUp(self) -> None:
        self.mm = MemoryManager(total_pages=10)

    def test_allocation_and_free(self) -> None:
        pages = self.mm.allocate(4)
        self.assertEqual(len(pages), 4)
        self.assertTrue(all(self.mm._bitmap[i] for i in pages))
        self.mm.free(pages)
        self.assertTrue(all(not self.mm._bitmap[i] for i in pages))

    def test_over_allocation(self) -> None:
        self.mm.allocate(10)
        with self.assertRaises(MemoryError):
            self.mm.allocate(1)

    def test_double_free(self) -> None:
        pages = self.mm.allocate(2)
        self.mm.free(pages)
        with self.assertRaises(ValueError):
            self.mm.free(pages)

    def test_invalid_page_index(self) -> None:
        with self.assertRaises(ValueError):
            self.mm.free([11])  # out of range

    def test_used_and_free_pages(self) -> None:
        a = self.mm.allocate(3)
        b = self.mm.allocate(2)
        used = self.mm.used_pages()
        free = self.mm.free_pages()
        self.assertCountEqual(used, a + b)
        self.assertCountEqual(free, [i for i in range(10) if i not in used])


if __name__ == '__main__':
    unittest.main()
