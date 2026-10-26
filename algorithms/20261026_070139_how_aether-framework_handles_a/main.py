import unittest
from core import ProcessMemory, PAGE_SIZE

class TestProcessMemory(unittest.TestCase):
    def setUp(self):
        # Reset global registry for each test
        global _page_registry, _page_id_counter
        _page_registry.clear()
        _page_id_counter = 0

    def test_allocate_and_read(self):
        proc = ProcessMemory()
        pid = proc.allocate_page("data1")
        self.assertEqual(proc.read(pid), "data1")
        self.assertEqual(proc.memory_usage(), PAGE_SIZE)

    def test_fork_shares_pages(self):
        parent = ProcessMemory()
        pid = parent.allocate_page("shared")
        child = parent.fork()
        self.assertEqual(parent.memory_usage(), PAGE_SIZE)
        self.assertEqual(child.memory_usage(), PAGE_SIZE)
        self.assertEqual(parent.shared_pages(), 1)
        self.assertEqual(child.shared_pages(), 1)
        self.assertEqual(parent.page_refcount(pid), 2)

    def test_child_write_triggers_copy(self):
        parent = ProcessMemory()
        pid = parent.allocate_page("orig")
        child = parent.fork()
        child.write(pid, "child")
        self.assertEqual(child.read(pid), "child")
        self.assertEqual(parent.read(pid), "orig")
        self.assertEqual(parent.memory_usage(), PAGE_SIZE)
        self.assertEqual(child.memory_usage(), PAGE_SIZE)
        self.assertEqual(parent.page_refcount(pid), 1)
        self.assertEqual(child.shared_pages(), 0)

    def test_parent_write_triggers_copy(self):
        parent = ProcessMemory()
        pid = parent.allocate_page("orig")
        child = parent.fork()
        parent.write(pid, "parent")
        self.assertEqual(parent.read(pid), "parent")
        self.assertEqual(child.read(pid), "orig")
        self.assertEqual(parent.page_refcount(pid), 1)
        self.assertEqual(child.shared_pages(), 0)

    def test_multiple_forks(self):
        parent = ProcessMemory()
        pid = parent.allocate_page("shared")
        child1 = parent.fork()
        child2 = parent.fork()
        self.assertEqual(parent.page_refcount(pid), 3)
        self.assertEqual(child1.page_refcount(pid), 3)
        self.assertEqual(child2.page_refcount(pid), 3)
        child1.write(pid, "c1")
        self.assertEqual(parent.read(pid), "shared")
        self.assertEqual(child1.read(pid), "c1")
        self.assertEqual(child2.read(pid), "shared")
        self.assertEqual(parent.page_refcount(pid), 2)
        self.assertEqual(child1.page_refcount(pid), 1)
        self.assertEqual(child2.page_refcount(pid), 2)

    def test_write_new_page(self):
        proc = ProcessMemory()
        pid = proc.allocate_page("initial")
        new_pid = 999  # arbitrary unused ID
        proc.write(new_pid, "new")
        self.assertEqual(proc.read(new_pid), "new")
        self.assertEqual(proc.memory_usage(), 2 * PAGE_SIZE)

    def test_memory_usage_after_multiple_writes(self):
        proc = ProcessMemory()
        pids = [proc.allocate_page(f"data{i}") for i in range(5)]
        self.assertEqual(proc.memory_usage(), 5 * PAGE_SIZE)
        proc.write(pids[0], "modified")
        self.assertEqual(proc.memory_usage(), 5 * PAGE_SIZE)

    def test_shared_pages_count(self):
        parent = ProcessMemory()
        pids = [parent.allocate_page(f"d{i}") for i in range(3)]
        child = parent.fork()
        self.assertEqual(parent.shared_pages(), 3)
        self.assertEqual(child.shared_pages(), 3)
        child.write(pids[1], "child")
        self.assertEqual(parent.shared_pages(), 2)
        self.assertEqual(child.shared_pages(), 1)

if __name__ == "__main__":
    # Demo: simulate fork with large memory
    parent = ProcessMemory()
    # Simulate allocating 10,000 pages (~40 MB)
    for i in range(10000):
        parent.allocate_page(f"page_{i}")
    print(f"Parent memory usage: {parent.memory_usage() / (1024*1024):.2f} MB")
    child = parent.fork()
    print(f"After fork, child memory usage: {child.memory_usage() / (1024*1024):.2f} MB")
    # Write to child to trigger copy
    child.write(1, "modified in child")
    print(f"After child write, parent memory usage: {parent.memory_usage() / (1024*1024):.2f} MB")
    print(f"After child write, child memory usage: {child.memory_usage() / (1024*1024):.2f} MB")
    # Run unit tests
    unittest.main(exit=False)
