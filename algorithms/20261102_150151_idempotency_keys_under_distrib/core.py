"""Idempotency key management under distributed retry pressure.

Provides deterministic key scoping, a bounded LRU storage contract with
TTL semantics, and a replay-safe execution guard.

Time invariants:
  - Key generation: O(len(scope))
  - Store lookup/insert: O(1) amortized (LRU + dict)
  - Eviction: O(1) amortized
Space invariants:
  - Store: O(capacity)
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class KeyStatus(Enum):
    """Lifecycle state of an idempotency key."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class IdempotencyKey:
    """Immutable, scoped idempotency key.

    The key is derived deterministically from (tenant, operation, payload_hash)
    so that retries with identical logical input produce the same key.
    """
    tenant: str
    operation: str
    payload_hash: str
    created_at: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f"{self.tenant}:{self.operation}:{self.payload_hash}"

    @property
    def fingerprint(self) -> str:
        """Stable 16-hex-char fingerprint for logging (not for identity)."""
        raw = f"{self.tenant}|{self.operation}|{self.payload_hash}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]


def scope_key(
    tenant: str,
    operation: str,
    payload: Dict[str, Any],
    *,
    salt: str = "",
) -> IdempotencyKey:
    """Derive a deterministic IdempotencyKey from logical scope.

    The payload is canonicalized (sorted keys, JSON-like serialization)
    before hashing so that key order does not affect identity.

    Args:
        tenant: Logical tenant / namespace.
        operation: Operation identifier (e.g. "charge", "transfer").
        payload: Logical request payload (must be JSON-serializable).
        salt: Optional secret salt to prevent cross-tenant key guessing.

    Returns:
        A frozen IdempotencyKey.

    Raises:
        ValueError: If tenant or operation is empty.
    """
    if not tenant or not tenant.strip():
        raise ValueError("tenant must be a non-empty string")
    if not operation or not operation.strip():
        raise ValueError("operation must be a non-empty string")

    canonical = _canonicalize(payload)
    digest_input = f"{tenant}|{operation}|{canonical}|{salt}".encode("utf-8")
    payload_hash = hashlib.sha256(digest_input).hexdigest()
    return IdempotencyKey(
        tenant=tenant,
        operation=operation,
        payload_hash=payload_hash,
    )


def _canonicalize(obj: Any) -> str:
    """Deterministic serialization for hashing.

    Uses sorted keys for dicts, tuple for lists, and repr for scalars.
    This is NOT a general JSON encoder; it is a stable canonical form
    sufficient for idempotency scoping.
    """
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        return repr(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return "{" + ",".join(f"{_canonicalize(k)}:{_canonicalize(v)}" for k, v in items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canonicalize(v) for v in obj) + "]"
    return repr(obj)


@dataclass
class KeyRecord:
    """Stored state for a single idempotency key."""
    key: IdempotencyKey
    status: KeyStatus
    result: Any = None
    error: Optional[str] = None
    stored_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: float("inf"))


class IdempotencyStore:
    """Bounded LRU store with TTL for idempotency key records.

    Contract:
      - At most `capacity` records are retained.
      - Records older than `ttl_seconds` are treated as expired on access.
      - Eviction is LRU: least-recently-used record is dropped first.
      - Thread-safety is NOT provided; callers must serialize access
        (e.g., via a single-threaded event loop or external lock).

    Time: O(1) amortized for get/put/evict.
    Space: O(capacity).
    """

    def __init__(self, capacity: int = 1024, ttl_seconds: float = 3600.0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._records: "OrderedDict[str, KeyRecord]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def _key_str(self, key: IdempotencyKey) -> str:
        return str(key)

    def get(self, key: IdempotencyKey) -> Optional[KeyRecord]:
        """Retrieve a live record, or None if absent/expired.

        On hit, the record is moved to MRU position.
        """
        ks = self._key_str(key)
        rec = self._records.get(ks)
        if rec is None:
            self._misses += 1
            return None
        if time.time() > rec.expires_at:
            # Expired: remove and count as miss
            del self._records[ks]
            self._misses += 1
            return None
        self._records.move_to_end(ks)
        self._hits += 1
        return rec

    def put(self, key: IdempotencyKey, status: KeyStatus,
            result: Any = None, error: Optional[str] = None) -> KeyRecord:
        """Insert or update a record. Evicts LRU if over capacity."""
        ks = self._key_str(key)
        now = time.time()
        rec = KeyRecord(
            key=key,
            status=status,
            result=result,
            error=error,
            stored_at=now,
            expires_at=now + self._ttl,
        )
        if ks in self._records:
            self._records.move_to_end(ks)
        self._records[ks] = rec
        while len(self._records) > self._capacity:
            self._records.popitem(last=False)
            self._evictions += 1
        return rec

    def peek(self, key: IdempotencyKey) -> Optional[KeyRecord]:
        """Non-mutating lookup (does not affect LRU order)."""
        ks = self._key_str(key)
        rec = self._records.get(ks)
        if rec is None:
            return None
        if time.time() > rec.expires_at:
            return None
        return rec

    def clear(self) -> None:
        self._records.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: IdempotencyKey) -> bool:
        return self.get(key) is not None


class IdempotencyGuard:
    """Replay-safe execution guard built on IdempotencyStore.

    Usage:
        guard = IdempotencyGuard(store)
        result = guard.execute(key, fn, *args, **kwargs)

    Semantics:
      - If key is absent: execute fn, store result as COMPLETED.
      - If key is COMPLETED: return stored result (replay).
      - If key is PENDING: raise ConcurrentExecutionError (another
        in-flight request holds the key).
      - If key is FAILED: re-execute fn (retry allowed on failure).

    Time: O(1) amortized for store ops + O(fn) for execution.
    """

    def __init__(self, store: IdempotencyStore) -> None:
        self._store = store

    def execute(
        self,
        key: IdempotencyKey,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        existing = self._store.get(key)
        if existing is not None:
            if existing.status == KeyStatus.COMPLETED:
                return existing.result
            if existing.status == KeyStatus.PENDING:
                raise ConcurrentExecutionError(
                    f"Key {key.fingerprint} is in-flight; concurrent execution detected"
                )
            # FAILED: fall through to re-execute

        # Mark as PENDING before execution to prevent concurrent duplicates
        self._store.put(key, KeyStatus.PENDING)
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self._store.put(key, KeyStatus.FAILED, error=str(exc))
            raise
        self._store.put(key, KeyStatus.COMPLETED, result=result)
        return result

    def is_replay(self, key: IdempotencyKey) -> bool:
        """Check whether a key would be served from cache (COMPLETED)."""
        rec = self._store.peek(key)
        return rec is not None and rec.status == KeyStatus.COMPLETED


class ConcurrentExecutionError(Exception):
    """Raised when a PENDING key is encountered (duplicate in-flight)."""
    pass
