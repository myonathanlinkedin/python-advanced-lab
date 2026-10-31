from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class HeaderField:
    """Immutable representation of an HTTP/2 header field."""
    name: str
    value: str

    @property
    def size(self) -> int:
        """
        HPACK size of a header field as defined by the spec:
        size = len(name) + len(value) + 32
        """
        return len(self.name) + len(self.value) + 32

    def __repr__(self) -> str:
        return f"HeaderField(name={self.name!r}, value={self.value!r})"


class DynamicTable:
    """
    HPACK dynamic table with a configurable maximum size.
    New entries are inserted at the beginning (index 1) and older entries are evicted
    when the size limit would be exceeded.
    """

    def __init__(self, max_size: int = 4096) -> None:
        if max_size < 0:
            raise ValueError("max_size must be non‑negative")
        self.max_size: int = max_size
        self.entries: List[HeaderField] = []
        self.current_size: int = 0

    def _evict_until_fit(self, needed: int) -> None:
        """Evict oldest entries until `needed` bytes can be accommodated."""
        while self.entries and self.current_size + needed > self.max_size:
            evicted = self.entries.pop()  # remove oldest (end of list)
            self.current_size -= evicted.size

    def add(self, header: HeaderField) -> None:
        """Add a header to the dynamic table, evicting as necessary."""
        if header.size > self.max_size:
            # Header too large to fit; clear table per spec.
            self.clear()
            return
        self._evict_until_fit(header.size)
        self.entries.insert(0, header)
        self.current_size += header.size

    def get(self, index: int) -> HeaderField:
        """
        Retrieve a header by 1‑based index as defined by HPACK.
        Index 1 refers to the most recent entry.
        """
        if index < 1 or index > len(self.entries):
            raise IndexError("Dynamic table index out of range")
        return self.entries[index - 1]

    def clear(self) -> None:
        """Remove all entries."""
        self.entries.clear()
        self.current_size = 0

    def __len__(self) -> int:
        return len(self.entries)


# Minimal static table (subset of the official HPACK static table) for demonstration.
_STATIC_TABLE: List[HeaderField] = [
    HeaderField(":authority", ""),
    HeaderField(":method", "GET"),
    HeaderField(":method", "POST"),
    HeaderField(":path", "/"),
    HeaderField(":path", "/index.html"),
    HeaderField(":scheme", "http"),
    HeaderField(":scheme", "https"),
    HeaderField(":status", "200"),
    HeaderField(":status", "204"),
    HeaderField(":status", "206"),
    HeaderField(":status", "304"),
    HeaderField(":status", "400"),
    HeaderField(":status", "404"),
    HeaderField(":status", "500"),
    HeaderField("accept-charset", ""),
    HeaderField("accept-encoding", "gzip, deflate"),
    HeaderField("accept-language", ""),
    HeaderField("accept-ranges", ""),
    HeaderField("accept", ""),
    HeaderField("access-control-allow-origin", ""),
    # ... the real table contains 61 entries; this subset suffices for tests.
]


class HPACKEncoder:
    """
    Simplified HPACK encoder.
    For each header it attempts to emit an indexed representation (static or dynamic).
    If not found, it emits a literal and adds the header to the dynamic table.
    The encoded form is a list of tuples:
        (index, None)          – indexed header (index > 0)
        (0, HeaderField)       – literal header (index == 0)
    """

    def __init__(self, dynamic_table: Optional[DynamicTable] = None) -> None:
        self.dynamic_table = dynamic_table or DynamicTable()
        self.static_table = _STATIC_TABLE

    def _find_index(self, header: HeaderField) -> Tuple[int, bool]:
        """
        Return (index, is_static) where index is 1‑based within the chosen table.
        If not found, returns (0, False).
        """
        # Search static table first.
        for i, entry in enumerate(self.static_table, start=1):
            if entry == header:
                return i, True
        # Search dynamic table.
        for i, entry in enumerate(self.dynamic_table.entries, start=1):
            if entry == header:
                return i, False
        return 0, False

    def encode(self, headers: List[HeaderField]) -> List[Tuple[int, Optional[HeaderField]]]:
        encoded: List[Tuple[int, Optional[HeaderField]]] = []
        for header in headers:
            idx, is_static = self._find_index(header)
            if idx:
                # Indexed representation.
                encoded.append((idx, None))
            else:
                # Literal representation; add to dynamic table.
                encoded.append((0, header))
                self.dynamic_table.add(header)
        return encoded


class HPACKDecoder:
    """
    Simplified HPACK decoder matching the encoder's format.
    It reconstructs the original header list and updates its own dynamic table.
    """

    def __init__(self, dynamic_table: Optional[DynamicTable] = None) -> None:
        self.dynamic_table = dynamic_table or DynamicTable()
        self.static_table = _STATIC_TABLE

    def decode(self, encoded: List[Tuple[int, Optional[HeaderField]]]) -> List[HeaderField]:
        decoded: List[HeaderField] = []
        for idx, literal in encoded:
            if idx == 0:
                if literal is None:
                    raise ValueError("Literal entry missing HeaderField")
                decoded.append(literal)
                self.dynamic_table.add(literal)
            else:
                # Resolve index: first static, then dynamic.
                if idx <= len(self.static_table):
                    header = self.static_table[idx - 1]
                else:
                    dyn_idx = idx - len(self.static_table)
                    header = self.dynamic_table.get(dyn_idx)
                decoded.append(header)
        return decoded


class MemorySavingsCalculator:
    """
    Utility to compute naive vs. encoded memory consumption.
    The model is deliberately simple:
        - Naive size: sum(len(name) + len(value)) for each header.
        - Encoded size:
            * Indexed header: 2 bytes (index representation).
            * Literal header: len(name) + len(value) + 2 bytes overhead.
    """

    @staticmethod
    def naive_size(headers: List[HeaderField]) -> int:
        return sum(len(h.name) + len(h.value) for h in headers)

    @staticmethod
    def encoded_size(encoded: List[Tuple[int, Optional[HeaderField]]]) -> int:
        size = 0
        for idx, literal in encoded:
            if idx == 0:
                # literal
                if literal is None:
                    raise ValueError("Literal entry missing HeaderField")
                size += len(literal.name) + len(literal.value) + 2
            else:
                # indexed
                size += 2
        return size

    @classmethod
    def savings(cls, headers: List[HeaderField], encoded: List[Tuple[int, Optional[HeaderField]]]) -> int:
        return cls.naive_size(headers) - cls.encoded_size(encoded)
