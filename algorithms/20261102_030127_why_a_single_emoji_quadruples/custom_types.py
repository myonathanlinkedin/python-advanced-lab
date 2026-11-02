from dataclasses import dat
from typing import Final


@dataclass(frozen=True)
class AetherString:
    """
    Immutable representation of a string in the Aether interpreter.

    The interpreter stores strings as UTF‑8 encoded byte sequences.
    Memory usage is therefore the length of that byte sequence.
    """
    raw: str

    @property
    def utf8_bytes(self) -> bytes:
        """UTF‑8 encoding of the underlying Python string."""
        return self.raw.encode("utf-8")

    def memory_bytes(self) -> int:
        """Number of bytes required to store the string in memory."""
        return len(self.utf8_bytes)
