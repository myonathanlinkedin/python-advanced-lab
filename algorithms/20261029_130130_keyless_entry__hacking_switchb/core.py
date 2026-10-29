"""
core.py
---------

Pure Python implementation of a simulated SwitchBot smart lock and a deterministic
brute‑force attack algorithm.

The module provides:
* `SwitchBotLock` – a lock that stores a secret numeric PIN.
* `generate_random_pin` – helper to create a random PIN of a given length.
* `brute_force_lock` – deterministic attacker that discovers the PIN by exhaustive
  search over the numeric space.

All code uses only the Python standard library and includes full type hints.
"""

from __future__ import annotations

import itertools
import random
import string
from typing import Final


# --------------------------------------------------------------------------- #
# Public constants
# --------------------------------------------------------------------------- #
DIGITS: Final = string.digits  # "0123456789"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class InvalidPinError(ValueError):
    """Raised when a PIN does not satisfy the lock's constraints."""


# --------------------------------------------------------------------------- #
# Core lock implementation
# --------------------------------------------------------------------------- #
class SwitchBotLock:
    """
    Simulated SwitchBot smart lock.

    The lock stores a secret numeric PIN.  The only public operation is
    `unlock(attempt)`, which returns ``True`` iff the supplied attempt matches
    the secret exactly.

    The class is deliberately immutable after construction – the secret cannot
    be read or altered, mirroring the behaviour of a real hardware lock.
    """

    __slots__ = ("_pin",)

    def __init__(self, pin: str) -> None:
        """
        Initialise the lock with a secret PIN.

        Parameters
        ----------
        pin: str
            A non‑empty string consisting solely of decimal digits.

        Raises
        ------
        InvalidPinError
            If ``pin`` is empty or contains non‑digit characters.
        """
        if not pin:
            raise InvalidPinError("PIN must be non‑empty.")
        if any(ch not in DIGITS for ch in pin):
            raise InvalidPinError("PIN must contain only decimal digits.")
        self._pin: str = pin

    def unlock(self, attempt: str) -> bool:
        """
        Attempt to unlock the lock.

        Parameters
        ----------
        attempt: str
            The candidate PIN.

        Returns
        -------
        bool
            ``True`` if ``attempt`` matches the secret PIN, ``False`` otherwise.
        """
        return attempt == self._pin

    def get_pin_length(self) -> int:
        """
        Return the length of the secret PIN.

        Returns
        -------
        int
            Number of digits in the secret PIN.
        """
        return len(self._pin)


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def generate_random_pin(length: int) -> str:
    """
    Generate a cryptographically strong random numeric PIN.

    Parameters
    ----------
    length: int
        Desired length of the PIN; must be a positive integer.

    Returns
    -------
    str
        Random PIN consisting of ``length`` decimal digits.

    Raises
    ------
    ValueError
        If ``length`` is not positive.
    """
    if length <= 0:
        raise ValueError("PIN length must be a positive integer.")
    # Using SystemRandom for cryptographic quality randomness.
    rng = random.SystemRandom()
    return "".join(rng.choice(DIGITS) for _ in range(length))


# --------------------------------------------------------------------------- #
# Brute‑force attacker
# --------------------------------------------------------------------------- #
def brute_force_lock(lock: SwitchBotLock) -> str:
    """
    Deterministically discover the secret PIN of ``lock`` by exhaustive search.

    The algorithm iterates over the Cartesian product of ``DIGITS`` repeated
    ``N`` times, where ``N`` is the length of the secret PIN.  The first candidate
    that unlocks the lock is returned.

    Time Complexity
    ----------------
    O(10^N) attempts, where N = lock.get_pin_length().

    Space Complexity
    ----------------
    O(1) auxiliary space (iteration is lazy).

    Parameters
    ----------
    lock: SwitchBotLock
        The lock whose secret PIN is to be discovered.

    Returns
    -------
    str
        The discovered PIN.  The function is guaranteed to terminate because
        the search space contains the true secret.

    Raises
    ------
    RuntimeError
        If the lock cannot be opened after exhausting the entire search space.
        (This would indicate a logical inconsistency.)
    """
    pin_len = lock.get_pin_length()
    # itertools.product yields tuples lazily; converting to string on the fly.
    for candidate_tuple in itertools.product(DIGITS, repeat=pin_len):
        candidate = "".join(candidate_tuple)
        if lock.unlock(candidate):
            return candidate
    raise RuntimeError("Failed to brute‑force the lock – inconsistent state.")


# End of core.py
