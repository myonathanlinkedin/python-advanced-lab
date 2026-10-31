import unittest
from typing import List, Tuple

from core import MemoryPool, measure_memory_savings, InvalidHandleError, DoubleFreeError


class TestMemoryPool(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = MemoryPool()

    def test_basic_allocation(self) -> None:
        h1 = self.pool.allocate(64)
        h2 = self.pool.allocate(128)
        self.assertNotEqual(h1, h2)
        self.assertEqual(self.pool.current_memory_in_use(), 192)
        self.assertEqual(self.pool.system_memory(), 192)

    def test_reuse_after_free(self) -> None:
        h1 = self.pool.allocate(32)
        self.pool.free(h1)
        # System memory should still be 32 after free
        self.assertEqual(self.pool.system_memory(), 32)
        # Allocate same size again – should reuse handle without extra system memory
        h2 = self.pool.allocate(32)
        self.assertEqual(self.pool.system_memory(), 32)
        self.assertEqual(self.pool.current_memory_in_use(), 32)
        self.assertNotEqual(h1, h2)  # handle may differ; reuse is about memory, not handle

    def test_multiple_sizes_and_reuse(self) -> None:
        handles = [self.pool.allocate(s) for s in (16, 32, 16, 64, 32)]
        for h in handles[:3]:
            self.pool.free(h)
        # At this point, free lists contain sizes 16 (2 blocks) and 32 (1 block)
        self.assertEqual(self.pool.system_memory(), sum([16, 32, 16, 64, 32]))
        # Allocate a 16 and a 32 – should reuse without extra system memory
        h_new1 = self.pool.allocate(16)
        h_new2 = self.pool.allocate(32)
        self.assertEqual(self.pool.system_memory(), sum([16, 32, 16, 64, 32]))
        self.assertEqual(self.pool.current_memory_in_use(), 16 + 32 + 64 + 32 + 16)

    def test_invalid_free(self) -> None:
        with self.assertRaises(InvalidHandleError):
            self.pool.free(999)

    def test_double_free(self) -> None:
        h = self.pool.allocate(10)
        self.pool.free(h)
        with self.assertRaises(InvalidHandleError):
            self.pool.free(h)

    def test_stats(self) -> None:
        h1 = self.pool.allocate(20)
        h2 = self.pool.allocate(30)
        self.pool.free(h1)
        sys_mem, in_use, saved = self.pool.stats()
        self.assertEqual(sys_mem, 50)
        self.assertEqual(in_use, 30)
        self.assertEqual(saved, 20)


class TestMeasureMemorySavings(unittest.TestCase):
    def run_sequence(self, seq: List[int]) -> Tuple[int, int, int]:
        return measure_memory_savings(seq)

    def test_simple_sequence(self) -> None:
        # Allocate 10, 20, then free 20, then allocate 20 again (reuse)
        seq = [10, 20, -20, 20]
        sys_mem, in_use, saved = self.run_sequence(seq)
        self.assertEqual(sys_mem, 30)   # 10 + 20 (first two allocations)
        self.assertEqual(in_use, 30)    # 10 + 20 (final state)
        self.assertEqual(saved, 0)      # No net saving because 20 was reallocated

    def test_savings(self) -> None:
        # Allocate 5, 5, free both, then allocate 5 again (reuse both)
        seq = [5, 5, -5, -5, 5]
        sys_mem, in_use, saved = self.run_sequence(seq)
        self.assertEqual(sys_mem, 10)   # only first two allocations needed
        self.assertEqual(in_use, 5)
        self.assertEqual(saved, 5)      # one block reused

    def test_complex_pattern(self) -> None:
        seq = [8, 16, 8, -8, 32, -16, 16, -8, 8]
        sys_mem, in_use, saved = self.run_sequence(seq)
        # First allocations: 8+16+8+32 = 64
        self.assertEqual(sys_mem, 64)
        # Final allocated sizes: 32 + 16 + 8 = 56
        self.assertEqual(in_use, 56)
        self.assertEqual(saved, 8)

    def test_invalid_free(self) -> None:
        seq = [10, -20]  # trying to free size that was never allocated
        with self.assertRaises(InvalidHandleError):
            self.run_sequence(seq)


def benchmark_pool() -> None:
    """
    Simple benchmark demonstrating memory savings on a repetitive pattern.
    Not part of unit tests; executed only when running the module directly.
    """
    import time

    pattern = [64, 128, 256, -64, -128, 64, 128, -256, 64]
    repetitions = 100_000

    start = time.perf_counter()
    for _ in range(repetitions):
        measure_memory_savings(pattern)
    elapsed = time.perf_counter() - start
    print(f"Benchmark completed in {elapsed:.3f}s for {repetitions} repetitions.")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demonstration
    print("\n--- Demonstration of MemoryPool ---")
    demo_pool = MemoryPool()
    h_a = demo_pool.allocate(100)
    h_b = demo_pool.allocate(200)
    print(f"System memory after allocations: {demo_pool.system_memory()} bytes")
    demo_pool.free(h_a)
    print(f"Memory saved after freeing one block: {demo_pool.memory_saved()} bytes")
    h_c = demo_pool.allocate(100)  # should reuse memory of h_a
    print(f"System memory after reallocation: {demo_pool.system_memory()} bytes")
    print(f"Current in‑use memory: {demo_pool.current_memory_in_use()} bytes")
    print(f"Final stats: {demo_pool.stats()}")

    # Run a quick benchmark
    print("\n--- Running quick benchmark ---")
    benchmark_pool()
