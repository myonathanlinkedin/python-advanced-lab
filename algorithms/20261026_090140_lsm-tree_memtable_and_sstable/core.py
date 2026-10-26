import os
import bisect
from pathlib import Path
from typing import List, Tuple, Optional, Iterator, Dict


class MemTable:
    """In‑memory sorted map with O(log n) insert/delete/search."""

    def __init__(self) -> None:
        self._values: Dict[bytes, Optional[bytes]] = {}
        self._keys: List[bytes] = []

    def put(self, key: bytes, value: bytes) -> None:
        if key not in self._values:
            bisect.insort(self._keys, key)
        self._values[key] = value

    def delete(self, key: bytes) -> None:
        # Store a tombstone (None) to indicate deletion.
        if key not in self._values:
            bisect.insort(self._keys, key)
        self._values[key] = None

    def get(self, key: bytes) -> Optional[bytes]:
        return self._values.get(key, None)

    def __len__(self) -> int:
        return len(self._values)

    def iterator(self) -> Iterator[Tuple[bytes, Optional[bytes]]]:
        for k in self._keys:
            yield k, self._values[k]

    def clear(self) -> None:
        self._values.clear()
        self._keys.clear()


class SSTable:
    """Immutable on‑disk sorted table. Simple line‑based format: <key>\\t<value>."""

    _SEP = b'\t'

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: List[Tuple[bytes, Optional[bytes]]] = []
        self._load()

    @staticmethod
    def _encode(key: bytes, value: Optional[bytes]) -> bytes:
        if value is None:
            # Represent tombstone as empty value.
            return key + SSTable._SEP + b'\n'
        return key + SSTable._SEP + value + b'\n'

    @staticmethod
    def _decode(line: bytes) -> Tuple[bytes, Optional[bytes]]:
        key, val = line.rstrip(b'\n').split(SSTable._SEP, 1)
        if val == b'':
            return key, None
        return key, val

    def _load(self) -> None:
        with self.path.open('rb') as f:
            for line in f:
                key, val = self._decode(line)
                self._entries.append((key, val))

    @classmethod
    def create_from_memtable(cls, mem: MemTable, path: Path) -> 'SSTable':
        # Write sorted entries to disk.
        with path.open('wb') as f:
            for key, val in mem.iterator():
                f.write(cls._encode(key, val))
        return cls(path)

    def get(self, key: bytes) -> Optional[bytes]:
        # Binary search over the in‑memory index.
        lo, hi = 0, len(self._entries) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_key = self._entries[mid][0]
            if mid_key == key:
                return self._entries[mid][1]
            elif mid_key < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def iterator(self) -> Iterator[Tuple[bytes, Optional[bytes]]]:
        return iter(self._entries)


class LSMTree:
    """Very small LSM‑Tree with a single mutable MemTable and two on‑disk levels."""

    def __init__(
        self,
        directory: Path,
        memtable_threshold: int = 4,
        level0_max: int = 2,
    ) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

        self.mem = MemTable()
        self.memtable_threshold = memtable_threshold
        self.level0_max = level0_max

        self.level0: List[SSTable] = []          # newest at end
        self.level1: Optional[SSTable] = None    # single merged run

        self._seq_counter = 0  # for unique file names

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def put(self, key: bytes, value: bytes) -> None:
        self.mem.put(key, value)
        if len(self.mem) >= self.memtable_threshold:
            self._flush_memtable()

    def delete(self, key: bytes) -> None:
        self.mem.delete(key)
        if len(self.mem) >= self.memtable_threshold:
            self._flush_memtable()

    def get(self, key: bytes) -> Optional[bytes]:
        # 1. MemTable (most recent)
        val = self.mem.get(key)
        if val is not None:
            return val
        if key in self.mem._values and self.mem._values[key] is None:
            return None

        # 2. Level‑0 (newest first)
        for sst in reversed(self.level0):
            val = sst.get(key)
            if val is not None:
                return val
            # tombstone handling
            if val is None and any(k == key for k, v in sst.iterator() if v is None):
                return None

        # 3. Level‑1
        if self.level1:
            return self.level1.get(key)
        return None

    # ------------------------------------------------------------------ #
    # Internal mechanics
    # ------------------------------------------------------------------ #
    def _next_path(self) -> Path:
        self._seq_counter += 1
        return self.dir / f'sstable_{self._seq_counter}.sst'

    def _flush_memtable(self) -> None:
        if len(self.mem) == 0:
            return
        path = self._next_path()
        sst = SSTable.create_from_memtable(self.mem, path)
        self.level0.append(sst)
        self.mem.clear()
        if len(self.level0) > self.level0_max:
            self._compact_level0_to_level1()

    def _compact_level0_to_level1(self) -> None:
        # Merge all level‑0 runs and the existing level‑1 (if any).
        iterators: List[Iterator[Tuple[bytes, Optional[bytes]]]] = [
            sst.iterator() for sst in self.level0
        ]
        if self.level1:
            iterators.append(self.level1.iterator())

        merged: List[Tuple[bytes, Optional[bytes]]] = self._merge_sorted(iterators)

        # Write new level‑1 file.
        level1_path = self.dir / 'level1.sst'
        with level1_path.open('wb') as f:
            for key, val in merged:
                f.write(SSTable._encode(key, val))

        # Replace old level‑1.
        self.level1 = SSTable(level1_path)

        # Delete old level‑0 files.
        for sst in self.level0:
            try:
                sst.path.unlink()
            except FileNotFoundError:
                pass
        self.level0.clear()

    @staticmethod
    def _merge_sorted(
        iterators: List[Iterator[Tuple[bytes, Optional[bytes]]]]
    ) -> List[Tuple[bytes, Optional[bytes]]]:
        """K‑way merge preserving the most recent value per key.
        The iterator order reflects recency: earlier iterators are newer.
        """
        # Load all entries into a list then sort; for tiny data sets this is fine.
        all_entries: List[Tuple[bytes, Optional[bytes], int]] = []
        for priority, it in enumerate(iterators):
            for key, val in it:
                all_entries.append((key, val, priority))

        # Sort by key, then by priority (lower priority = newer)
        all_entries.sort(key=lambda x: (x[0], x[2]))

        result: List[Tuple[bytes, Optional[bytes]]] = []
        seen: set = set()
        for key, val, _ in all_entries:
            if key in seen:
                continue
            seen.add(key)
            if val is not None:
                result.append((key, val))
            # If val is None (tombstone) we drop the key entirely.
        return result
