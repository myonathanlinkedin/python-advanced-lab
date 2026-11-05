from __future__ import annotations
import threading
import time
import random

from .engine import IdempotencyKeyManager, DuplicateKeyError, KeyNotFoundError
from .types import KeyScope, KeyStatus


def _demo() -> None:
    """
    Simple demonstration of the IdempotencyKeyManager usage.
    """
    manager = IdempotencyKeyManager(ttl_seconds=5.0)
    scope = KeyScope(name="user", identifier="alice")
    key = "op-123"

    # Acquire key
    record = manager.acquire(key, scope)
    print(f"Acquired key: {record}")

    # Simulate operation
    time.sleep(1)
    manager.release(key, scope, result={"status": "ok"})

    status = manager.get_status(key, scope)
    print(f"Key status after release: {status}")


def _test_duplicate_key() -> None:
    manager = IdempotencyKeyManager(ttl_seconds=10.0)
    scope = KeyScope(name="resource", identifier="res-1")
    key = "unique-op"

    # First acquisition should succeed
    manager.acquire(key, scope)

    # Second acquisition should raise DuplicateKeyError
    try:
        manager.acquire(key, scope)
        assert False, "DuplicateKeyError not raised"
    except DuplicateKeyError:
        pass  # Expected

    # Release and then acquire again
    manager.release(key, scope, result="done")
    manager.acquire(key, scope)  # Should succeed


def _test_expiration() -> None:
    manager = IdempotencyKeyManager(ttl_seconds=1.0)
    scope = KeyScope(name="session", identifier="sess-1")
    key = "temp-op"

    manager.acquire(key, scope)
    time.sleep(1.5)  # Wait for expiration
    try:
        manager.get_status(key, scope)
        assert False, "KeyNotFoundError not raised after expiration"
    except KeyNotFoundError:
        pass  # Expected


def _test_scoping() -> None:
    manager = IdempotencyKeyManager(ttl_seconds=10.0)
    key = "shared-op"

    scope_a = KeyScope(name="user", identifier="alice")
    scope_b = KeyScope(name="user", identifier="bob")

    manager.acquire(key, scope_a)
    manager.acquire(key, scope_b)  # Should succeed because scopes differ

    status_a = manager.get_status(key, scope_a)
    status_b = manager.get_status(key, scope_b)
    assert status_a == KeyStatus.PENDING
    assert status_b == KeyStatus.PENDING

    manager.release(key, scope_a, result="alice")
    manager.release(key, scope_b, result="bob")

    assert manager.get_status(key, scope_a) == KeyStatus.SUCCESS
    assert manager.get_status(key, scope_b) == KeyStatus.SUCCESS


def _test_concurrency() -> None:
    manager = IdempotencyKeyManager(ttl_seconds=5.0)
    scope = KeyScope(name="batch", identifier="batch-1")
    key = "concurrent-op"

    def worker(thread_id: int) -> None:
        try:
            manager.acquire(key, scope)
            # Simulate work
            time.sleep(random.uniform(0.01, 0.1))
            manager.release(key, scope, result=f"thread-{thread_id}")
        except DuplicateKeyError:
            # Expected for all but one thread
            pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Only one successful acquisition should exist
    status = manager.get_status(key, scope)
    assert status == KeyStatus.SUCCESS, f"Unexpected status: {status}"


def _run_all_tests() -> None:
    _demo()
    _test_duplicate_key()
    _test_expiration()
    _test_scoping()
    _test_concurrency()
    print("All tests passed.")


if __name__ == "__main__":
    _run_all_tests()
