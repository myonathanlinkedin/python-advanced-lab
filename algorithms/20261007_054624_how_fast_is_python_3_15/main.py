import unittest
from typing import List

from core import (
    BenchmarkResult,
    benchmark_loop,
    benchmark_function_calls,
    benchmark_list_comprehension,
    benchmark_dict_lookup,
    benchmark_string_concatenation,
    run_all_benchmarks,
)


class TestBenchmarks(unittest.TestCase):
    def _assert_positive(self, value: float) -> None:
        self.assertIsInstance(value, float)
        self.assertGreater(value, 0.0, "Duration should be positive")

    def test_loop(self) -> None:
        dur = benchmark_loop(1_000)
        self._assert_positive(dur)

    def test_function_calls(self) -> None:
        dur = benchmark_function_calls(1_000)
        self._assert_positive(dur)

    def test_list_comprehension(self) -> None:
        dur = benchmark_list_comprehension(1_000)
        self._assert_positive(dur)

    def test_dict_lookup(self) -> None:
        dur = benchmark_dict_lookup(1_000)
        self._assert_positive(dur)

    def test_string_concatenation(self) -> None:
        dur = benchmark_string_concatenation(500)
        self._assert_positive(dur)

    def test_run_all(self) -> None:
        results: List[BenchmarkResult] = run_all_benchmarks(1_000_000)
        self.assertEqual(len(results), 5)
        for res in results:
            with self.subTest(name=res.name):
                self._assert_positive(res.duration)
                self.assertIsInstance(res.ops_per_sec, float)
                self.assertGreater(res.ops_per_sec, 0.0)


def _demo() -> None:
    """Simple demonstration printing benchmark results."""
    print("Running Python 3.15 micro‑benchmarks (demo)...")
    results = run_all_benchmarks(5_000_000)
    for r in results:
        print(f"{r.name:20s} : {r.duration:8.4f}s  →  {r.ops_per_sec:10.2f} ops/s")


if __name__ == "__main__":
    # Run unit tests first; if they pass, show the demo.
    unittest.main(exit=False)
    _demo()
