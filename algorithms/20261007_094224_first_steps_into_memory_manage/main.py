import unittest
import time
from core import MemoryManager, MemoryHandle, WeakMemoryHandle


class TestMemoryManager(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = MemoryManager()

    def test_allocate_and_get(self) -> None:
        h = self.mgr.allocate(42)
        self.assertEqual(h.get(), 42)
        self.assertEqual(self.mgr._refcount(h._obj_id), 1)

    def test_set_value(self) -> None:
        h = self.mgr.allocate("a")
        h.set("b")
        self.assertEqual(h.get(), "b")

    def test_manual_refcount(self) -> None:
        h = self.mgr.allocate([1, 2])
        self.assertEqual(self.mgr._refcount(h._obj_id), 1)
        h.retain()
        self.assertEqual(self.mgr._refcount(h._obj_id), 2)
        h.release()
        self.assertEqual(self.mgr._refcount(h._obj_id), 1)

    def test_context_manager(self) -> None:
        h = self.mgr.allocate(0)
        with h as ctx:
            self.assertIs(ctx, h)
            self.assertEqual(self.mgr._refcount(h._obj_id), 2)
        self.assertEqual(self.mgr._refcount(h._obj_id), 1)

    def test_automatic_cleanup(self) -> None:
        h = self.mgr.allocate("temp")
        obj_id = h._obj_id
        self.assertIn(obj_id, self.mgr._store)
        del h
        # Force GC to trigger __del__
        import gc
        gc.collect()
        self.assertNotIn(obj_id, self.mgr._store)

    def test_weak_handle(self) -> None:
        h = self.mgr.allocate(99)
        w = h.weak()
        self.assertTrue(w.is_alive())
        self.assertEqual(w.get(), 99)
        del h
        import gc
        gc.collect()
        self.assertFalse(w.is_alive())
        with self.assertRaises(ReferenceError):
            _ = w.get()

    def test_multiple_handles(self) -> None:
        h1 = self.mgr.allocate("x")
        h2 = MemoryHandle(self.mgr, h1._obj_id)  # manual second handle
        self.assertEqual(self.mgr._refcount(h1._obj_id), 2)
        h1.release()
        self.assertEqual(self.mgr._refcount(h1._obj_id), 1)
        h2.release()
        self.assertNotIn(h1._obj_id, self.mgr._store)


def benchmark_allocation(iterations: int = 100_000) -> float:
    mgr = MemoryManager()
    start = time.perf_counter()
    for i in range(iterations):
        h = mgr.allocate(i)
        h.set(i * 2)
        h.release()
    return time.perf_counter() - start


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Simple demo
    mgr = MemoryManager()
    handle = mgr.allocate({"key": "value"})
    print("Initial:", handle.get())
    handle.set({"key": "new"})
    print("Updated:", handle.get())

    # Benchmark
    secs = benchmark_allocation(200_000)
    print(f"Benchmark: allocated 200k objects in {secs:.4f}s")
