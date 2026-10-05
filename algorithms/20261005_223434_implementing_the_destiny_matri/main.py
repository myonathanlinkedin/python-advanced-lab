"""
Entry point, unit tests, and simple benchmark for the Destiny Matrix implementation.
"""

import timeit
import unittest
from datetime import date

from core import (
    MAJOR_ARCANA,
    destiny_number,
    digit_sum,
    get_major_arcana,
    reduce_to_range,
)


class TestCoreFunctions(unittest.TestCase):
    def test_digit_sum(self):
        self.assertEqual(digit_sum(0), 0)
        self.assertEqual(digit_sum(7), 7)
        self.assertEqual(digit_sum(12345), 15)
        self.assertEqual(digit_sum(99999), 45)

        with self.assertRaises(ValueError):
            digit_sum(-1)

    def test_reduce_to_range(self):
        self.assertEqual(reduce_to_range(0), 0)
        self.assertEqual(reduce_to_range(21), 21)
        self.assertEqual(reduce_to_range(22), 4)   # 2+2=4
        self.assertEqual(reduce_to_range(123456), 6)  # 1+2+3+4+5+6=21 -> 21 in range
        self.assertEqual(reduce_to_range(999999), 9)  # 9+9+9+9+9+9=54 -> 5+4=9

        with self.assertRaises(ValueError):
            reduce_to_range(10, low=5, high=3)

    def test_destiny_number_known_values(self):
        # Known reductions:
        # 1990-12-31 -> 19901231 -> digit sum 1+9+9+0+1+2+3+1 = 26 -> 2+6 = 8
        self.assertEqual(destiny_number(date(1990, 12, 31)), 8)
        # 2000-01-01 -> 20000101 -> sum 2+0+0+0+0+1+0+1 = 4
        self.assertEqual(destiny_number(date(2000, 1, 1)), 4)
        # 1975-07-04 -> 19750704 -> sum 1+9+7+5+0+7+0+4 = 33 -> 3+3 = 6
        self.assertEqual(destiny_number(date(1975, 7, 4)), 6)

    def test_get_major_arcana(self):
        # Using the same dates as above, verify mapping to names.
        self.assertEqual(get_major_arcana(date(1990, 12, 31)), MAJOR_ARCANA[8])
        self.assertEqual(get_major_arcana(date(2000, 1, 1)), MAJOR_ARCANA[4])
        self.assertEqual(get_major_arcana(date(1975, 7, 4)), MAJOR_ARCANA[6])

    def test_invalid_date_type(self):
        with self.assertRaises(AttributeError):
            # Passing a string should raise because .year attribute is missing
            destiny_number("1990-12-31")  # type: ignore[arg-type]

    def test_boundary_dates(self):
        # Earliest representable date in datetime
        earliest = date.min
        latest = date.max
        # Ensure function runs without error and returns a valid index
        self.assertIn(destiny_number(earliest), range(22))
        self.assertIn(destiny_number(latest), range(22))


def benchmark_destiny_number(iterations: int = 100_000) -> float:
    """
    Simple benchmark for ``destiny_number`` using a representative date.
    Returns the elapsed time in seconds.
    """
    stmt = "destiny_number(date(1988, 5, 23))"
    setup = "from core import destiny_number; from datetime import date"
    return timeit.timeit(stmt, setup=setup, number=iterations)


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Demo
    demo_date = date.today()
    arcana = get_major_arcana(demo_date)
    print(f"Today's date: {demo_date.isoformat()}")
    print(f"Destiny Matrix number: {destiny_number(demo_date)}")
    print(f"Corresponding Major Arcana: {arcana}")

    # Benchmark
    elapsed = benchmark_destiny_number(200_000)
    print(f"\nBenchmark: 200,000 iterations took {elapsed:.4f} seconds "
          f"({elapsed/200_000:.8f} s per call)")
