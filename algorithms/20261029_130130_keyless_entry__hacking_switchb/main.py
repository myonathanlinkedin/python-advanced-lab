"""
main.py
---------

Unit‑test suite and demonstration entry‑point for the SwitchBot lock simulation.

The tests cover:
* Correct handling of valid and invalid PINs.
* Successful and failed unlock attempts.
* The deterministic brute‑force attacker on various PIN lengths.

Running this module directly executes the test suite.
"""

import unittest

from core import (
    SwitchBotLock,
    generate_random_pin,
    brute_force_lock,
    InvalidPinError,
)


class TestSwitchBotLock(unittest.TestCase):
    """Test cases for the SwitchBotLock and associated utilities."""

    def test_unlock_success(self) -> None:
        lock = SwitchBotLock("1234")
        self.assertTrue(lock.unlock("1234"))

    def test_unlock_failure(self) -> None:
        lock = SwitchBotLock("9876")
        self.assertFalse(lock.unlock("1234"))
        self.assertFalse(lock.unlock(""))  # empty attempt must fail

    def test_invalid_pin_initialisation(self) -> None:
        with self.assertRaises(InvalidPinError):
            SwitchBotLock("")  # empty PIN
        with self.assertRaises(InvalidPinError):
            SwitchBotLock("12a4")  # non‑digit character

    def test_generate_random_pin(self) -> None:
        pin = generate_random_pin(6)
        self.assertEqual(len(pin), 6)
        self.assertTrue(pin.isdigit())

        # Repeated calls should produce different values (high probability)
        pin2 = generate_random_pin(6)
        self.assertNotEqual(pin, pin2)

    def test_brute_force_small_pin(self) -> None:
        # Exhaustive search on a 2‑digit PIN – trivial runtime.
        secret = "07"
        lock = SwitchBotLock(secret)
        discovered = brute_force_lock(lock)
        self.assertEqual(discovered, secret)

    def test_brute_force_single_digit(self) -> None:
        for digit in "0", "5", "9":
            lock = SwitchBotLock(digit)
            self.assertEqual(brute_force_lock(lock), digit)

    def test_brute_force_random_pin(self) -> None:
        # Verify that the attacker works for a randomly generated 4‑digit PIN.
        secret = generate_random_pin(4)
        lock = SwitchBotLock(secret)
        discovered = brute_force_lock(lock)
        self.assertEqual(discovered, secret)


if __name__ == "__main__":
    # Run the unit‑test suite.
    unittest.main()
