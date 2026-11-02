"""Unit tests, benchmarks, and demo for idempotency key system."""
from __future__ import annotations

import time
import unittest
from typing import Any, Callable, Dict

from core import (
    ConcurrentExecutionError,
    IdempotencyGuard,
    IdempotencyKey,
    IdempotencyStore,
    KeyStatus,
    scope_key,
)


class TestKeyScoping(unittest.TestCase):
    """Deterministic key derivation and scoping."""

    def test_same_payload_same_key(self) -> None:
        k1 = scope_key("tenant-a", "charge", {"amount": 100, "currency": "USD"})
        k2 = scope_key("tenant-a", "charge", {"currency": "USD", "amount": 100})
        self.assertEqual(k1, k2)

    def test_different_tenant_different_key(self) -> None:
        k1 = scope_key("tenant-a", "charge", {"amount": 100})
        k2 = scope_key("tenant-b", "charge", {"amount": 100})
        self.assertNotEqual(k1, k2)

    def test_different_operation_different_key(self) -> None:
        k1 = scope_key("t", "charge", {"amount": 100})
        k2 = scope_key("t", "refund", {"amount": 100})
        self.assertNotEqual(k1, k2)

    def test_different_payload_different_key(self) -> None:
        k1 = scope_key("t", "op", {"x": 1})
        k2 = scope_key("t", "op", {"x": 2})
        self.assertNotEqual(k1, k2)

    def test_empty_tenant_raises(self) -> None:
        with self.assertRaises(ValueError):
            scope_key("", "op", {})

    def test_empty_operation_raises(self) -> None:
        with self.assertRaises(ValueError):
            scope_key("t", "", {})

    def test_nested_payload_canonical(self) -> None:
        p1 = {"a": [1, 2], "b": {"x": True, "y": None}}
        p2 = {"b": {"y": None, "x": True}, "a": [1, 2]}
        k1 = scope_key("t", "op", p1)
        k2 = scope_key("t", "op", p2)
        self.assertEqual(k1, k2)

    def test_salt_changes_key(self) -> None:
        k1 = scope_key("t", "op", {"v": 1}, salt="s1")
        k2 = scope_key("t", "op", {"v": 1}, salt="s2")
        self.assertNotEqual(k1, k2)

    def test_fingerprint_stable(self) -> None:
        k = scope_key("t", "op", {"v": 1})
        self.assertEqual(k.fingerprint, k.fingerprint)
        self.assertEqual(len(k.fingerprint), 16)


class TestStore(unittest.TestCase):
    """LRU + TTL store contract."""

    def test_put_get(self) -> None:
        s = IdempotencyStore(capacity=10)
        k = scope_key("t", "op", {"v": 1})
        s.put(k, KeyStatus.COMPLETED, result=42)
        rec = s.get(k)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.result, 42)
        self.assertEqual(rec.status, KeyStatus.COMPLETED)

    def test_miss_returns_none(self) -> None:
        s = IdempotencyStore(capacity=10)
        k = scope_key("t", "op", {"v": 999})
        self.assertIsNone(s.get(k))

    def test_lru_eviction(self) -> None:
        s = IdempotencyStore(capacity=3)
        keys = [scope_key("t", f"op{i}", {"i": i}) for i in range(5)]
        for k in keys:
            s.put(k, KeyStatus.COMPLETED, result=k.fingerprint)
        # First two should be evicted
        self.assertIsNone(s.get(keys[0]))
        self.assertIsNone(s.get(keys[1]))
        self.assertIsNotNone(s.get(keys[2]))
        self.assertEqual(s.size, 3)

    def test_lru_order_update_on_get(self) -> None:
        s = IdempotencyStore(capacity=3)
        k0 = scope_key("t", "op0", {"i": 0})
        k1 = scope_key("t", "op1", {"i": 1})
        k2 = scope_key("t", "op2", {"i": 2})
        for k in (k0, k1, k2):
            s.put(k, KeyStatus.COMPLETED)
        # Access k0 to make it MRU
        s.get(k0)
        # Insert k3 -> evicts k1 (LRU)
        k3 = scope_key("t", "op3", {"i": 3})
        s.put(k3, KeyStatus.COMPLETED)
        self.assertIsNotNone(s.get(k0))
        self.assertIsNone(s.get(k1))
        self.assertIsNotNone(s.get(k2))
        self.assertIsNotNone(s.get(k3))

    def test_ttl_expiry(self) -> None:
        s = IdempotencyStore(capacity=10, ttl_seconds=0.05)
        k = scope_key("t", "op", {"v": 1})
        s.put(k, KeyStatus.COMPLETED, result="x")
        self.assertIsNotNone(s.get(k))
        time.sleep(0.06)
        self.assertIsNone(s.get(k))

    def test_capacity_validation(self) -> None:
        with self.assertRaises(ValueError):
            IdempotencyStore(capacity=0)

    def test_ttl_validation(self) -> None:
        with self.assertRaises(ValueError):
            IdempotencyStore(ttl_seconds=0)

    def test_contains(self) -> None:
        s = IdempotencyStore(capacity=10)
        k = scope_key("t", "op", {"v": 1})
        self.assertNotIn(k, s)
        s.put(k, KeyStatus.COMPLETED)
        self.assertIn(k, s)

    def test_hit_rate(self) -> None:
        s = IdempotencyStore(capacity=10)
        k = scope_key("t", "op", {"v": 1})
        s.put(k, KeyStatus.COMPLETED)
        s.get(k)   # hit
        s.get(scope_key("t", "op", {"v": 2}))  # miss
        self.assertAlmostEqual(s.hit_rate, 0.5)


class TestGuard(unittest.TestCase):
    """Replay-safe execution semantics."""

    def test_first_execution_runs_fn(self) -> None:
        s = IdempotencyStore(capacity=10)
        g = IdempotencyGuard(s)
        k = scope_key("t", "op", {"v": 1})
        calls = []
        result = g.execute(k, lambda: calls.append(1) or 99)
        self.assertEqual(result, 99)
        self.assertEqual(len(calls), 1)

    def test_replay_returns_cached(self) -> None:
        s = IdempotencyStore(capacity=10)
        g = IdempotencyGuard(s)
        k = scope_key("t", "op", {"v": 1})
        calls = []
        g.execute(k, lambda: calls.append(1) or 42)
        result = g.execute(k, lambda: calls.append(2) or 999)
        self.assertEqual(result, 42)
        self.assertEqual(len(calls), 1)  # fn not re-executed

    def test_failed_key_allows_retry(self) -> None:
        s = IdempotencyStore(capacity=10)
        g = IdempotencyGuard(s)
        k = scope_key("t", "op", {"v": 1})

        def flaky() -> int:
            flaky.count = getattr(flaky, "count", 0) + 1
            if flaky.count == 1:
                raise RuntimeError("transient")
            return 7

        with self.assertRaises(RuntimeError):
            g.execute(k, flaky)
        result = g.execute(k, flaky)
        self.assertEqual(result, 7)
        self.assertEqual(flaky.count, 2)

    def test_concurrent_pending_raises(self) -> None:
        s = IdempotencyStore(capacity=10)
        g = IdempotencyGuard(s)
        k = scope_key("t", "op", {"v": 1})
        # Simulate in-flight by inserting PENDING directly
        s.put(k, KeyStatus.PENDING)
        with self.assertRaises(ConcurrentExecutionError):
            g.execute(k, lambda: 1)

    def test_is_replay(self) -> None:
        s = IdempotencyStore(capacity=10)
        g = IdempotencyGuard(s)
        k = scope_key("t", "op", {"v": 1})
        self.assertFalse(g.is_replay(k))
        g.execute(k, lambda: 1)
        self.assertTrue(g.is_replay(k))


class TestBenchmark(unittest.TestCase):
    """Deterministic performance sanity checks."""

    def test_throughput_10k_ops(self) -> None:
        s = IdempotencyStore(capacity=5000)
        g = IdempotencyGuard(s)
        n = 10_000
        start = time.perf_counter()
        for i in range(n):
            k = scope_key("bench", "op", {"i": i % 5000})
            g.execute(k, lambda i=i: i * 2)
        elapsed = time.perf_counter() - start
        ops_per_sec = n / elapsed
        # Sanity: should be well above 10k ops/sec on any modern machine
        self.assertGreater(ops_per_sec, 10_000)
        self.assertLess(elapsed, 2.0)


def _demo() -> None:
    """Runnable demo showing key scoping, replay, and retry."""
    print("=== Idempotency Key Demo ===")
    store = IdempotencyStore(capacity=100, ttl_seconds=60)
    guard = IdempotencyGuard(store)

    # Simulate a payment charge
    key = scope_key("acme", "charge", {"amount": 49.99, "currency": "USD"})
    print(f"Key: {key}")
    print(f"Fingerprint: {key.fingerprint}")

    def charge(amount: float) -> Dict[str, Any]:
        return {"status": "charged", "amount": amount, "txn_id": "TXN-001"}

    # First call executes
    r1 = guard.execute(key, charge, 49.99)
    print(f"1st call: {r1}")

    # Retry (same logical request) -> replay
    r2 = guard.execute(key, charge, 49.99)
    print(f"2nd call (replay): {r2}")
    print(f"Same result: {r1 == r2}")
    print(f"Store size: {len(store)}, Hit rate: {store.hit_rate:.2f}")
    print("=== Demo Complete ===")


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
    _demo()
