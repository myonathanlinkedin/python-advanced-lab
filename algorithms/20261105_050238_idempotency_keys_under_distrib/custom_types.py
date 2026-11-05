from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
import time


class KeyStatus(Enum):
    """Enumeration of possible idempotency key states."""
    PENDING = auto()
    SUCCESS = auto()
    FAILURE = auto()
    EXPIRED = auto()


@dataclass(frozen=True)
class KeyScope:
    """
    Represents a logical scope for an idempotency key.
    Scopes can be used to partition keys, e.g., per user or per resource.
    """
    name: str
    identifier: str

    def __hash__(self) -> int:
        return hash((self.name, self.identifier))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, KeyScope):
            return False
        return (self.name, self.identifier) == (other.name, other.identifier)


@dataclass
class IdempotencyKey:
    """
    Represents an idempotency key record.
    """
    key: str
    scope: KeyScope
    status: KeyStatus = KeyStatus.PENDING
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[Exception] = None

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        if now is None:
            now = time.time()
        return now >= self.expires_at
