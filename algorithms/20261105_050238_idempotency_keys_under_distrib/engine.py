from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import threading
import time

from .types import IdempotencyKey, KeyScope, KeyStatus


class IdempotencyKeyManager:
    """
    Thread-safe manager for idempotency keys.
    Supports key acquisition, release, status querying, and automatic expiration cleanup.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        """
        :param ttl_seconds: Default time-to-live for keys in seconds.
        """
        self._ttl = ttl_seconds
        self._store: Dict[Tuple[str, KeyScope], IdempotencyKey] = {}
        self._lock = threading.RLock()

    def _now(self) -> float:
        return time.time()

    def acquire(self, key: str, scope: KeyScope, ttl: Optional[float] = None) -> IdempotencyKey:
        """
        Acquire an idempotency key. If the key already exists and is not expired,
        raises a DuplicateKeyError. Otherwise, creates a new key record.
        """
        with self._lock:
            self.cleanup_expired()
            composite = (key, scope)
            existing = self._store.get(composite)
            if existing:
                if existing.is_expired(self._now()):
                    # Remove expired key and proceed
                    del self._store[composite]
                else:
                    raise DuplicateKeyError(f"Key '{key}' in scope '{scope}' already exists.")
            expires_at = self._now() + (ttl if ttl is not None else self._ttl)
            record = IdempotencyKey(key=key, scope=scope, expires_at=expires_at)
            self._store[composite] = record
            return record

    def release(self, key: str, scope: KeyScope, result: Any = None, error: Optional[Exception] = None) -> None:
        """
        Release an idempotency key after operation completion.
        Sets status to SUCCESS or FAILURE based on presence of error.
        """
        with self._lock:
            composite = (key, scope)
            record = self._store.get(composite)
            if not record:
                raise KeyNotFoundError(f"Key '{key}' in scope '{scope}' not found for release.")
            if record.is_expired(self._now()):
                # If expired, treat as failure
                record.status = KeyStatus.EXPIRED
                record.error = Exception("Key expired before release.")
                del self._store[composite]
                return
            if error:
                record.status = KeyStatus.FAILURE
                record.error = error
            else:
                record.status = KeyStatus.SUCCESS
                record.result = result
            # Keep the record until TTL expires; do not delete immediately
            # Optionally, could delete here if desired.

    def get_status(self, key: str, scope: KeyScope) -> KeyStatus:
        """
        Retrieve the current status of an idempotency key.
        """
        with self._lock:
            composite = (key, scope)
            record = self._store.get(composite)
            if not record:
                raise KeyNotFoundError(f"Key '{key}' in scope '{scope}' not found.")
            if record.is_expired(self._now()):
                record.status = KeyStatus.EXPIRED
                del self._store[composite]
                return KeyStatus.EXPIRED
            return record.status

    def cleanup_expired(self) -> None:
        """
        Remove all expired keys from the store.
        """
        with self._lock:
            now = self._now()
            expired_keys = [k for k, v in self._store.items() if v.is_expired(now)]
            for k in expired_keys:
                del self._store[k]

    def __len__(self) -> int:
        """
        Return the number of active (non-expired) keys.
        """
        with self._lock:
            self.cleanup_expired()
            return len(self._store)


class DuplicateKeyError(RuntimeError):
    """Raised when attempting to acquire a key that already exists and is not expired."""


class KeyNotFoundError(RuntimeError):
    """Raised when a key is not found in the store."""
