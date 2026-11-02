from typing import Any

from custom_types import AetherString


def memory_usage(s: AetherString) -> int:
    """
    Return the memory consumption (in bytes) of the given AetherString.

    This is a thin wrapper around AetherString.memory_bytes() to keep the
    public API stable.
    """
    return s.memory_bytes()


def append_char(s: AetherString, ch: str) -> AetherString:
    """
    Return a new AetherString with the character `ch` appended.

    The operation is pure – the original AetherString is left untouched.
    """
    if not isinstance(ch, str) or len(ch) != 1:
        raise ValueError("`ch` must be a single Unicode character.")
    return AetherString(s.raw + ch)


def compare_memory_growth(base: AetherString, added: AetherString) -> float:
    """
    Compute the factor by which memory grew when moving from `base` to `added`.

    Raises:
        ValueError: If the base string occupies zero bytes (division by zero).
    """
    base_bytes = base.memory_bytes()
    if base_bytes == 0:
        raise ValueError("Base string memory is zero; cannot compute growth factor.")
    return added.memory_bytes() / base_bytes
