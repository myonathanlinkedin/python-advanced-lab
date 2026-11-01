import random
import unittest
import time
from typing import List, Tuple

from core import CountMinSketch


class TestCountMinSketch(unittest.TestCase):
    def setUp(self) -> None:
        # Width and depth chosen to give reasonable error bounds for tests
        self.width = 2000
        self.depth = 5
        self.sketch = CountMinSketch(width=self.width, depth=self.depth, seed=42)

    def test_basic_estimation(self) -> None:
        # Insert known frequencies
        frequencies = {'a': 1200, 'b': 800, 'c': 50}
        for item, freq in frequencies.items():
            for _ in range(freq):
                self.sketch.update(item)

        # Verify estimates are never below true count
        for item, true_cnt in frequencies.items():
            est = self.sketch.estimate(item)
            self.assertGreaterEqual(est, true_cnt,
                f"Estimate for {item} below true count")

        # Verify error bound: ε = 2/width, N = total updates
        epsilon = 2.0 / self.width
        N = sum(frequencies.values())
        max_error = int(epsilon * N) + 1  # +1 for integer rounding safety
        for item, true_cnt in frequencies.items():
            est = self.sketch.estimate(item)
            self.assertLessEqual(est - true_cnt, max_error,
                f"Estimate error for {item} exceeds bound")

    def test_merge(self) -> None:
        # Two independent sketches
        sketch1 = CountMinSketch(width=self.width, depth=self.depth, seed=42)
        sketch2 = CountMinSketch(width=self.width, depth=self.depth, seed=42)

        # Populate with disjoint streams
        for _ in range(500):
            sketch1.update('x')
        for _ in range(300):
            sketch2.update('x')
        for _ in range(200):
            sketch1.update('y')
        for _ in range(400):
            sketch2.update('z')

        # Merge sketch2 into sketch1
        sketch1.merge(sketch2)

        # Expected true counts
        true_counts = {'x': 800, 'y': 200, 'z': 400}
        epsilon = 2.0 / self.width
        N = sum(true_counts.values())
        max_error = int(epsilon * N) + 1

        for item, true_cnt in true_counts.items():
            est = sketch1.estimate(item)
            self.assertGreaterEqual(est, true_cnt)
            self.assertLessEqual(est - true_cnt, max_error)

    def test_heavy_hitters(self) -> None:
        # Generate a stream with a few heavy items
        heavy_items = {'H1': 1500, 'H2': 1200}
        light_items = {f'L{i}': random.randint(1, 30) for i in range(50)}
        stream = []
        for item, cnt in heavy_items.items():
            stream.extend([item] * cnt)
        for item, cnt in light_items.items():
            stream.extend([item] * cnt)

        random.shuffle(stream)

        for itm in stream:
            self.sketch.update(itm)

        # Threshold set just below the smallest heavy count
        threshold = 1000
        hh = self.sketch.heavy_hitters(threshold)

        # Extract just the item identifiers
        hh_items = {itm for itm, _ in hh}
        self.assertTrue(heavy_items.keys() <= hh_items,
            "All heavy items should be reported")
        # Ensure no light item crosses the threshold
        for itm in hh_items:
            self.assertIn(itm, heavy_items,
                f"Light item {itm} incorrectly reported as heavy")

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            CountMinSketch(width=0, depth=5)
        with self.assertRaises(ValueError):
            CountMinSketch(width=10, depth=-1)

        with self.assertRaises(ValueError):
            self.sketch.update('test', increment=-3)

        with self.assertRaises(ValueError):
            self.sketch.heavy_hitters(-1)

        other = CountMinSketch(width=10, depth=5, seed=99)
        with self.assertRaises(ValueError):
            self.sketch.merge(other)


def benchmark_sketch(num_items: int = 1_000_000, width: int = 5000, depth: int = 7) -> Tuple[float, float]:
    """
    Simple benchmark: time to ingest *num_items* random integers and
    time to query a random subset.
    Returns (ingest_time_seconds, query_time_seconds).
    """
    sketch = CountMinSketch(width=width, depth=depth, seed=123)
    rng = random.Random(0)

    start = time.perf_counter()
    for _ in range(num_items):
        sketch.update(rng.randint(0, 10_000))
    ingest_time = time.perf_counter() - start

    queries = [rng.randint(0, 10_000) for _ in range(10_000)]
    start = time.perf_counter()
    for q in queries:
        _ = sketch.estimate(q)
    query_time = time.perf_counter() - start

    return ingest_time, query_time


if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Run a quick benchmark demonstration
    ing_time, qry_time = benchmark_sketch(num_items=200_000, width=3000, depth=6)
    print(f"Ingestion time for 200k items: {ing_time:.3f}s")
    print(f"Query time for 10k queries: {qry_time:.3f}s")
