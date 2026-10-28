from __future__ import annotations

import threading
import time

from .engine import BoundedBlockingQueue, Full, Empty

def _test_basic_put_get() -> None:
    q = BoundedBlockingQueue[int](3)
    assert q.empty()
    q.put(1)
    q.put(2)
    q.put(3)
    assert q.full()
    try:
        q.put(4, block=False)
        assert False, "Expected Full exception"
    except Full:
        pass

    assert q.get() == 1
    assert q.get() == 2
    assert q.get() == 3
    assert q.empty()
    try:
        q.get(block=False)
        assert False, "Expected Empty exception"
    except Empty:
        pass

def _test_blocking_put_timeout() -> None:
    q = BoundedBlockingQueue[int](1)
    q.put(42)  # fills the queue

    start = time.monotonic()
    try:
        q.put(99, timeout=0.2)
        assert False, "Expected Full exception after timeout"
    except Full:
        elapsed = time.monotonic() - start
        assert 0.15 <= elapsed <= 0.3, f"Timeout inaccurate: {elapsed}s"

def _test_blocking_get_timeout() -> None:
    q = BoundedBlockingQueue[int](2)

    start = time.monotonic()
    try:
        q.get(timeout=0.1)
        assert False, "Expected Empty exception after timeout"
    except Empty:
        elapsed = time.monotonic() - start
        assert 0.05 <= elapsed <= 0.2, f"Timeout inaccurate: {elapsed}s"

def _test_concurrent_producer_consumer() -> None:
    q = BoundedBlockingQueue[int](5)
    produced = []
    consumed = []

    def producer():
        for i in range(10):
            q.put(i)
            produced.append(i)
            time.sleep(0.01)  # simulate work

    def consumer():
        for _ in range(10):
            item = q.get()
            consumed.append(item)
            time.sleep(0.015)  # simulate work

    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)

    t_prod.start()
    t_cons.start()
    t_prod.join()
    t_cons.join()

    assert produced == list(range(10)), "Producer sequence corrupted"
    assert consumed == list(range(10)), "Consumer sequence corrupted"
    assert q.empty(), "Queue should be empty after balanced work"

def _run_all_tests() -> None:
    _test_basic_put_get()
    _test_blocking_put_timeout()
    _test_blocking_get_timeout()
    _test_concurrent_producer_consumer()
    print("All unit tests passed.")

if __name__ == '__main__':
    _run_all_tests()
    # Demonstration of live usage
    demo_q = BoundedBlockingQueue[str](2)

    def demo_producer():
        for word in ("alpha", "beta", "gamma"):
            print(f"Producer putting: {word}")
            demo_q.put(word)
            time.sleep(0.2)

    def demo_consumer():
        for _ in range(3):
            item = demo_q.get()
            print(f"Consumer got: {item}")
            time.sleep(0.4)

    threading.Thread(target=demo_producer, daemon=True).start()
    threading.Thread(target=demo_consumer, daemon=True).start()
    # Allow demo threads to finish
    time.sleep(2)
