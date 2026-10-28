"""Unit tests, benchmarks, and entrypoint for the Memory Pool Block Allocator."""
from __future__ import annotations

import time
import unittest

from core import Block, MemoryPool


class TestMemoryPoolInit(unittest.TestCase):
    def test_valid_init(self) -> None:
        pool = MemoryPool(pool_size=1024, block_size=64)
        self.assertEqual(pool.num_blocks, 16)
        self.assertEqual(pool.stats()["free_blocks"], 16)

    def test_invalid_pool_size(self) -> None:
        with self.assertRaises(ValueError):
            MemoryPool(pool_size=0, block_size=64)

    def test_invalid_block_size(self) -> None:
        with self.assertRaises(ValueError):
            MemoryPool(pool_size=1024, block_size=0)

    def test_non_multiple(self) -> None:
        with self.assertRaises(ValueError):
            MemoryPool(pool_size=100, block_size=64)

    def test_single_block(self) -> None:
        pool = MemoryPool(pool_size=64, block_size=64)
        self.assertEqual(pool.num_blocks, 1)


class TestAllocate(unittest.TestCase):
    def test_allocate_returns_block(self) -> None:
        pool = MemoryPool(pool_size=256, block_size=64)
        blk = pool.allocate()
        self.assertIsNotNone(blk)
        self.assertTrue(blk.is_allocated)
        self.assertEqual(pool.stats()["allocated_blocks"], 1)

    def test_allocate_exhaustion(self) -> None:
        pool = MemoryPool(pool_size=128, block_size=64)
        b1 = pool.allocate()
        b2 = pool.allocate()
        b3 = pool.allocate()
        self.assertIsNotNone(b1)
        self.assertIsNotNone(b2)
        self.assertIsNone(b3)

    def test_allocate_unique_indices(self) -> None:
        pool = MemoryPool(pool_size=512, block_size=64)
        indices = set()
        for _ in range(8):
            blk = pool.allocate()
            self.assertIsNotNone(blk)
            indices.add(blk.index)
        self.assertEqual(len(indices), 8)


class TestDeallocate(unittest.TestCase):
    def test_deallocate_returns_to_free(self) -> None:
        pool = MemoryPool(pool_size=256, block_size=64)
        blk = pool.allocate()
        pool.deallocate(blk)
        self.assertFalse(blk.is_allocated)
        self.assertEqual(pool.stats()["free_blocks"], 4)

    def test_deallocate_unallocated_raises(self) -> None:
        pool = MemoryPool(pool_size=256, block_size=64)
        blk = pool._blocks[0]  # not allocated
        with self.assertRaises(ValueError):
            pool.deallocate(blk)

    def test_deallocate_wrong_pool_raises(self) -> None:
        pool1 = MemoryPool(pool_size=256, block_size=64)
        pool2 = MemoryPool(pool_size=256, block_size=64)
        blk = pool1.allocate()
        with self.assertRaises(ValueError):
            pool2.deallocate(blk)

    def test_reallocate_after_deallocate(self) -> None:
        pool = MemoryPool(pool_size=128, block_size=64)
        b1 = pool.allocate()
        pool.deallocate(b1)
        b2 = pool.allocate()
        self.assertIsNotNone(b2)
        self.assertTrue(b2.is_allocated)


class TestStats(unittest.TestCase):
    def test_stats_consistency(self) -> None:
        pool = MemoryPool(pool_size=1024, block_size=64)
        blocks = [pool.allocate() for _ in range(5)]
        s = pool.stats()
        self.assertEqual(s["allocated_blocks"], 5)
        self.assertEqual(s["free_blocks"], 11)
        self.assertEqual(s["allocated_bytes"], 5 * 64)
        self.assertEqual(s["free_bytes"], 11 * 64)
        self.assertEqual(s["allocated_bytes"] + s["free_bytes"], 1024)

    def test_stats_after_full_cycle(self) -> None:
        pool = MemoryPool(pool_size=512, block_size=64)
        blocks = [pool.allocate() for _ in range(8)]
        for b in blocks:
            pool.deallocate(b)
        s = pool.stats()
        self.assertEqual(s["free_blocks"], 8)
        self.assertEqual(s["allocated_blocks"], 0)


class TestEdgeCases(unittest.TestCase):
    def test_single_block_pool(self) -> None:
        pool = MemoryPool(pool_size=64, block_size=64)
        b = pool.allocate()
        self.assertIsNotNone(b)
        self.assertIsNone(pool.allocate())
        pool.deallocate(b)
        b2 = pool.allocate()
        self.assertIsNotNone(b2)

    def test_many_alloc_dealloc_cycles(self) -> None:
        pool = MemoryPool(pool_size=4096, block_size=64)
        for _ in range(100):
            b = pool.allocate()
            self.assertIsNotNone(b)
            pool.deallocate(b)
        self.assertEqual(pool.stats()["free_blocks"], 64)


class TestBenchmarks(unittest.TestCase):
    def test_allocation_throughput(self) -> None:
        pool = MemoryPool(pool_size=1 << 20, block_size=64)  # 16384 blocks
        n = 10000
        start = time.perf_counter()
        for _ in range(n):
            blk = pool.allocate()
            if blk is not None:
                pool.deallocate(blk)
        elapsed = time.perf_counter() - start
        ops_per_sec = n / elapsed if elapsed > 0 else float("inf")
        # Should be well above 1M ops/sec on modern hardware.
        self.assertGreater(ops_per_sec, 100_000)


def _demo() -> None:
    """Run a small demonstration of the allocator."""
    pool = MemoryPool(pool_size=1024, block_size=64)
    print(f"Pool: {pool}")
    blocks = []
    for i in range(5):
        b = pool.allocate()
        blocks.append(b)
        print(f"  Allocated block index={b.index}")
    print(f"  Stats: {pool.stats()}")
    for b in blocks:
        pool.deallocate(b)
    print(f"  After dealloc: {pool.stats()}")


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
    _demo()
