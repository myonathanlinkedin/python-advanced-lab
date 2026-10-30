from __future__ import annotations
from dataclasses import dataclass
from bisect import bisect_left
from typing import Any, List, Tuple, Optional

@dataclass(frozen=True)
class SSTable:
    entries: List[Tuple[Any, Any]]  # sorted by key

    def get(self, key: Any) -> Optional[Any]:
        i = bisect_left(self.entries, (key, None))
        if i < len(self.entries) and self.entries[i][0] == key:
            return self.entries[i][1]
        return None

class MemTable:
    def __init__(self, max_size: int = 1000):
        self._data: dict[Any, Any] = {}
        self.max_size = max_size

    def put(self, key: Any, value: Any) -> None:
        self._data[key] = value

    def delete(self, key: Any) -> None:
        self._data[key] = None  # tombstone

    def get(self, key: Any) -> Optional[Any]:
        return self._data.get(key, None)

    def size(self) -> int:
        return len(self._data)

    def flush(self) -> SSTable:
        entries = sorted((k, v) for k, v in self._data.items() if v is not None)
        sstable = SSTable(entries)
        self._data.clear()
        return sstable

class LSMTree:
    def __init__(self, memtable_size: int = 1000, compaction_threshold: int = 4):
        self.memtable = MemTable(memtable_size)
        self.sstables: List[SSTable] = []
        self.compaction_threshold = compaction_threshold

    def put(self, key: Any, value: Any) -> None:
        self.memtable.put(key, value)
        if self.memtable.size() >= self.memtable.max_size:
            self.flush()

    def delete(self, key: Any) -> None:
        self.memtable.delete(key)
        if self.memtable.size() >= self.memtable.max_size:
            self.flush()

    def get(self, key: Any) -> Optional[Any]:
        val = self.memtable.get(key)
        if val is not None:
            return val
        for sstable in reversed(self.sstables):
            val = sstable.get(key)
            if val is not None:
                return val
        return None

    def flush(self) -> None:
        sstable = self.memtable.flush()
        if sstable.entries:
            self.sstables.append(sstable)
        if len(self.sstables) >= self.compaction_threshold:
            self.compact()

    def compact(self) -> None:
        merged: List[Tuple[Any, Any]] = []
        iterators = [iter(s.entries) for s in self.sstables]
        current = [next(it, None) for it in iterators]
        while any(entry is not None for entry in current):
            min_key = min(entry[0] for entry in current if entry is not None)
            values = [entry[1] for entry in current if entry is not None and entry[0] == min_key]
            value = values[-1]
            if value is not None:
                merged.append((min_key, value))
            for i, entry in enumerate(current):
                if entry is not None and entry[0] == min_key:
                    current[i] = next(iterators[i], None)
        self.sstables = [SSTable(merged)]

    def __len__(self) -> int:
        return len(self.sstables) + self.memtable.size()
