from __future__ import annotations
from fractions import Fraction
from typing import Dict, List, Tuple, Union


class TextCRDT:
    """
    A simple sequence CRDT for collaborative text editing.
    Uses rational numbers (Fraction) as position identifiers,
    guaranteeing infinite density between any two positions.
    """

    def __init__(self, site_id: str) -> None:
        self.site_id: str = site_id
        # Mapping from position (Fraction) to character
        self._store: Dict[Fraction, str] = {}
        # Counter for generating unique tie‑breakers in case of equal positions
        self._counter: int = 0

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def _sorted_positions(store: Dict[Fraction, str]) -> List[Fraction]:
        return sorted(store.keys())

    def _next_counter(self) -> int:
        self._counter += 1
        return self._counter

    # --------------------------------------------------------------------- #
    # Public API – local operations
    # --------------------------------------------------------------------- #
    def local_insert(self, index: int, char: str) -> Dict[str, Union[str, int, Fraction]]:
        """
        Insert `char` locally at `index`.
        Returns an operation dict that can be transmitted to remote replicas.
        """
        if not (0 <= index <= len(self)):
            raise IndexError("Insert index out of bounds")
        left_key = Fraction(0) if index == 0 else self._sorted_positions(self._store)[index - 1]
        right_key = Fraction(1) if index == len(self) else self._sorted_positions(self._store)[index]
        new_key = (left_key + right_key) / 2
        # Ensure uniqueness even if two inserts happen at the same rational point
        # (extremely unlikely but we guard with a tiny epsilon based on a counter)
        epsilon = Fraction(1, 10 ** (6 + self._next_counter()))
        new_key += epsilon
        self._store[new_key] = char
        return {"type": "insert", "key": new_key, "char": char, "site": self.site_id}

    def local_delete(self, index: int) -> Dict[str, Union[str, Fraction]]:
        """
        Delete character locally at `index`.
        Returns an operation dict for remote application.
        """
        if not (0 <= index < len(self)):
            raise IndexError("Delete index out of bounds")
        key = self._sorted_positions(self._store)[index]
        del self._store[key]
        return {"type": "delete", "key": key, "site": self.site_id}

    # --------------------------------------------------------------------- #
    # Public API – remote operations
    # --------------------------------------------------------------------- #
    def apply_remote(self, op: Dict[str, Union[str, Fraction]]) -> None:
        """
        Apply a remote operation (insert or delete) to this replica.
        The operation dict must contain the fields produced by the local methods.
        """
        op_type = op.get("type")
        if op_type == "insert":
            key: Fraction = op["key"]  # type: ignore
            char: str = op["char"]  # type: ignore
            # If the key already exists (duplicate delivery), ignore.
            if key not in self._store:
                self._store[key] = char
        elif op_type == "delete":
            key: Fraction = op["key"]  # type: ignore
            self._store.pop(key, None)  # safe delete
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")

    # --------------------------------------------------------------------- #
    # Inspection utilities
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._store)

    def __str__(self) -> str:
        ordered_keys = self._sorted_positions(self._store)
        return "".join(self._store[k] for k in ordered_keys)

    def to_list(self) -> List[Tuple[Fraction, str]]:
        """Return the internal state as a sorted list of (position, char)."""
        return [(k, self._store[k]) for k in self._sorted_positions(self._store)]
