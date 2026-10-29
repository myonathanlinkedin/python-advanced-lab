from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Dict, Optional, List

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')


@dataclass(frozen=True)
class CacheEntry(Generic[TKey, TValue]):
    """
    Immutable representation of a cache entry.
    """
    key: TKey
    value: TValue
