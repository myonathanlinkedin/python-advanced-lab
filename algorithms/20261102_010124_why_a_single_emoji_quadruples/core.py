"""
core.py
---------

Utility functions to analyse UTF‑8 memory consumption of Python strings,
illustrating why a single emoji can quadruple the memory footprint compared
to plain ASCII characters.

The module provides:
- `utf8_byte_length(s)`: exact byte count of `s` when encoded as UTF‑8.
- `is_emoji(ch)`: heuristic detection of emoji characters (code points > 0xFFFF).
- `memory_factor(s)`: ratio of UTF‑8 bytes to the number of ASCII characters,
  i.e. the factor by which memory grows relative to a pure‑ASCII string of the
  same length.
"""

from __future__ import annotations

from typing import Iterable


def utf8_byte_length(s: str) -> int:
    """
    Return the number of bytes required to store ``s`` in UTF‑8.

    The implementation uses Python's built‑in UTF‑8 encoder which follows the
    Unicode Standard precisely, guaranteeing a mathematically correct result.

    Parameters
    ----------
    s: str
        Input Unicode string.

    Returns
    -------
    int
        Length of ``s.encode('utf-8')`` in bytes.
    """
    return len(s.encode('utf-8'))


def is_emoji(ch: str) -> bool:
    """
    Heuristic test for whether a single Unicode character is an emoji.

    The Unicode Standard places most modern emoji outside the Basic Multilingual
    Plane (BMP), i.e. with code points greater than ``0xFFFF``.  While this is
    not a perfect test (some BMP symbols are also emoji), it is sufficient for
    demonstrating the memory‑size effect: any character with a code point > 0xFFFF
    is encoded in UTF‑8 using four bytes, which is exactly four times the size of
    a typical ASCII character.

    Parameters
    ----------
    ch: str
        A single Unicode character.

    Returns
    -------
    bool
        ``True`` if ``ord(ch) > 0xFFFF``; ``False`` otherwise.
    """
    if len(ch) != 1:
        raise ValueError("is_emoji expects a single character")
    return ord(ch) > 0xFFFF


def memory_factor(s: str) -> float:
    """
    Compute the memory expansion factor of ``s`` relative to an ASCII string
    of the same length.

    For a string consisting solely of ASCII characters, the factor is 1.0.
    For each emoji (or any character that requires four UTF‑8 bytes) the factor
    contributes an additional 3 bytes, raising the overall ratio.

    The function is mathematically defined as:

        factor = utf8_byte_length(s) / max(1, len(s))

    The ``max`` guard avoids division by zero for the empty string, returning
    ``0.0`` in that case.

    Parameters
    ----------
    s: str
        Input Unicode string.

    Returns
    -------
    float
        Ratio of UTF‑8 byte count to the number of characters (treated as ASCII).
    """
    n = len(s)
    if n == 0:
        return 0.0
    return utf8_byte_length(s) / n


def emoji_memory_impact(emojis: Iterable[str]) -> int:
    """
    Given an iterable of emoji characters, return the total extra bytes
    contributed beyond the baseline of one byte per character.

    For each emoji (four UTF‑8 bytes) the extra cost is three bytes.

    Parameters
    ----------
    emojis: Iterable[str]
        Iterable yielding single-character emoji strings.

    Returns
    -------
    int
        Sum of extra bytes over the ASCII baseline.
    """
    extra = 0
    for ch in emojis:
        if len(ch) != 1:
            raise ValueError("Each element must be a single character")
        extra += 3 if is_emoji(ch) else 0
    return extra
