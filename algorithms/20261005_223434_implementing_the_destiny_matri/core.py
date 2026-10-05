"""
Core implementation of the Destiny Matrix algorithm.

The algorithm reduces a birth date to a single integer in the range 0‑21
(which corresponds to the 22 Major Arcana of the Tarot).
The reduction is performed by repeatedly summing the decimal digits of the
concatenated year, month, and day until the result falls within the target range.
"""

from __future__ import annotations

from datetime import date
from typing import List


# List of the 22 Major Arcana names, indexed 0‑21.
MAJOR_ARCANA: List[str] = [
    "The Fool", "The Magician", "The High Priestess", "The Empress",
    "The Emperor", "The Hierophant", "The Lovers", "The Chariot",
    "Strength", "The Hermit", "Wheel of Fortune", "Justice",
    "The Hanged Man", "Death", "Temperance", "The Devil",
    "The Tower", "The Star", "The Moon", "The Sun",
    "Judgement", "The World"
]


def digit_sum(n: int) -> int:
    """
    Return the sum of the decimal digits of a non‑negative integer.
    """
    if n < 0:
        raise ValueError("digit_sum expects a non‑negative integer")
    total = 0
    while n:
        total += n % 10
        n //= 10
    return total


def reduce_to_range(value: int, low: int = 0, high: int = 21) -> int:
    """
    Reduce ``value`` to the inclusive range [low, high] by repeatedly applying
    digit_sum.  The function assumes low <= high and that the range size is
    less than 100 (which holds for the Major Arcana case).
    """
    if low > high:
        raise ValueError("low must not exceed high")
    while not (low <= value <= high):
        value = digit_sum(value)
    return value


def _date_to_int_components(birth_date: date) -> List[int]:
    """
    Convert a :class:`datetime.date` into a list of its year, month, and day
    components as integers.
    """
    return [birth_date.year, birth_date.month, birth_date.day]


def _concatenate_components(components: List[int]) -> int:
    """
    Concatenate the decimal representations of the components into a single integer.
    Example: [1990, 12, 31] -> 19901231
    """
    return int("".join(f"{c:0>2}" for c in components))


def destiny_number(birth_date: date) -> int:
    """
    Compute the Destiny Matrix number for ``birth_date``.
    The result is an integer in the range 0‑21 inclusive.
    """
    components = _date_to_int_components(birth_date)
    concatenated = _concatenate_components(components)
    return reduce_to_range(concatenated, 0, 21)


def get_major_arcana(birth_date: date) -> str:
    """
    Return the Major Arcana name associated with ``birth_date``.
    """
    index = destiny_number(birth_date)
    return MAJOR_ARCANA[index]
