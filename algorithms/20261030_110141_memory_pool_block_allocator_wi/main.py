"""
Unit tests and simple demonstration for the MemoryPool allocator.
"""

import unittest
from core import MemoryPool, OutOfMemoryError, InvalidFreeError


class TestMemoryPool(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = MemoryPool(total_size=1024, block_size=64)  # 16 blocks

    def test_initial_state(self) -> None:
        self.assertEqual(self.pool.total_blocks, 16)
        self.assertEqual(self.pool.free_blocks, 16)
        self.assertEqual(self.pool.allocated_blocks, 0)

    def test_allocate_one(self) -> None:
        block = self.pool.allocate()
        self.assertEqual(len(block), 64)
        self.assertEqual(self.pool.free_blocks, 15)
        self.assertEqual(self.pool.allocated_blocks, 1)

        # Write and read back to ensure the view works.
        block[:4] = b"test"
        self.assertEqual(bytes(block[:4]), b"test")

    def test_allocate_all_and_exhaustion(self) -> None:
        blocks = [self.pool.allocate() for _ in range(self.pool.total_blocks)]
        self.assertEqual(self.pool.free_blocks, 0)
        self.assertEqual(self.pool.allocated_blocks, self.pool.total_blocks)

        with self.assertRaises(OutOfMemoryError):
            self.pool.allocate()

        # Free one block and allocate again.
        self.pool.free(blocks[0])
        self.assertEqual(self.pool.free_blocks, 1)
        new_block = self.pool.allocate()
        self.assertIsNotNone(new_block)
        self.assertEqual(self.pool.free_blocks, 0)

    def test_double_free_detection(self) -> None:
        block = self.pool.allocate()
        self.pool.free(block)
        with self.assertRaises(InvalidFreeError):
            self.pool.free(block)  # second free should fail

    def test_invalid_free(self) -> None:
        # Create a memoryview that does not belong to the pool.
        external = memoryview(bytearray(64))
        with self.assertRaises(InvalidFreeError):
            self.pool.free(external)

        # Allocate a block, then tamper with its hidden attribute.
        block = self.pool.allocate()
        delattr(block, "_pool_offset")  # type: ignore[attr-defined]
        with self.assertRaises(InvalidFreeError):
            self.pool.free(block)

    def test_thread_safety_basic(self) -> None:
        # Simple sanity check: allocate and free from multiple threads.
        import threading

        def worker(allocations: int) -> None:
            local_blocks = []
            for _ in range(allocations):
                try:
                    blk = self.pool.allocate()
                    local_blocks.append(blk)
                except OutOfMemoryError:
                    break
            for blk in local_blocks:
                self.pool.free(blk)

        threads = [threading.Thread(target=worker, args=(10,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After all threads finish, pool must be fully restored.
        self.assertEqual(self.pool.free_blocks, self.pool.total_blocks)
        self.assertEqual(self.pool.allocated_blocks, 0)

    def test_invalid_initialization(self) -> None:
        with self.assertRaises(ValueError):
            MemoryPool(total_size=100, block_size=0)
        with self.assertRaises(ValueError):
            MemoryPool(total_size=-1, block_size=10)
        with self.assertRaises(ValueError):
            MemoryPool(total_size=100, block_size=30)  # not a divisor


if __name__ == "__main__":
    # Run unit tests with verbose output.
    unittest.main(verbosity=2)

# Demo usage (executed only when run as script, not during import)
def demo() -> None:
    print("=== MemoryPool Demo ===")
    pool = MemoryPool(total_size=256, block_size=32)  # 8 blocks
    print(f"Total blocks: {pool.total_blocks}")
    print(f"Free blocks initially: {pool.free_blocks}")

    blk = pool.allocate()
    print(f"Allocated block of length {len(blk)}")
    blk[:] = b"Hello, MemoryPool!"
    print("Block contents:", bytes(blk[:20]))

    pool.free(blk)
    print(f"Free blocks after free: {pool.free_blocks}")


if __name__ == "__main__":
    demo()
