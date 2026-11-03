import unittest
import random
from typing import List, Tuple

from core import CountMinSketch


class TestCountMinSketch(unittest.TestCase):
    def test_basic_estimation(self) -> None:
        cms = CountMinSketch(width=1000, depth=5, seed=42)
        stream = ["apple", "banana", "apple", "cherry", "banana", "apple"]
        true_counts = {}
        for item in stream:
            cms.add(item)
            true_counts[item] = true_counts.get(item, 0) + 1

        for item, true_cnt in true_counts.items():
            est = cms.estimate(item)
            self.assertGreaterEqual(est, true_cnt,
                f"Estimate {est} should be >= true count {true_cnt} for {item}")
            # The error bound for this configuration is ε·N ≈ e/width * N
            # With width=1000 and N=6, ε·N < 0.02, so estimate must equal true count.
            self.assertEqual(est, true_cnt,
                f"With small N, estimate should match true count for {item}")

    def test_merge(self) -> None:
        cms1 = CountMinSketch(width=500, depth=4, seed=7)
        cms2 = CountMinSketch(width=500, depth=4, seed=7)

        stream1 = ["x", "y", "x", "z"]
        stream2 = ["y", "y", "w", "x"]

        for item in stream1:
            cms1.add(item)
        for item in stream2:
            cms2.add(item)

        cms1.merge(cms2)

        # Verify merged counts via estimate
        combined_counts = {}
        for item in stream1 + stream2:
            combined_counts[item] = combined_counts.get(item, 0) + 1

        for item, true_cnt in combined_counts.items():
            est = cms1.estimate(item)
            self.assertGreaterEqual(est, true_cnt)
            # With moderate width, the estimate should still be exact for this tiny N.
            self.assertEqual(est, true_cnt)

    def test_heavy_hitters_detection(self) -> None:
        # Parameters chosen to give reasonable error bounds.
        cms = CountMinSketch.with_error(epsilon=0.01, delta=0.001, seed=123)

        # Generate a stream where a few items are heavy.
        heavy_items = ["A", "B"]
        light_items = [f"L{i}" for i in range(100)]

        random.seed(0)
        stream: List[str] = []

        # Heavy items appear 500 times each.
        for _ in range(500):
            stream.append(random.choice(heavy_items))
        # Light items appear 5 times each.
        for item in light_items:
            for _ in range(5):
                stream.append(item)

        random.shuffle(stream)

        for item in stream:
            cms.add(item)

        # Threshold set just below the heavy count.
        threshold = 400
        hitters = cms.heavy_hitters(threshold)
        hitter_set = {item for item, _ in hitters}
        self.assertTrue(set(heavy_items).issubset(hitter_set),
            f"Heavy items {heavy_items} should be reported as hitters")
        # Ensure no light item is mistakenly reported.
        for item in light_items:
            self.assertNotIn(item, hitter_set,
                f"Light item {item} incorrectly reported as heavy")

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            CountMinSketch(width=0, depth=5)
        with self.assertRaises(ValueError):
            CountMinSketch(width=10, depth=-1)
        with self.assertRaises(ValueError):
            CountMinSketch.with_error(epsilon=1.5, delta=0.1)
        with self.assertRaises(ValueError):
            CountMinSketch.with_error(epsilon=0.1, delta=0.0)

    def test_merge_incompatible(self) -> None:
        cms_a = CountMinSketch(width=10, depth=3, seed=1)
        cms_b = CountMinSketch(width=10, depth=4, seed=1)  # different depth
        with self.assertRaises(ValueError):
            cms_a.merge(cms_b)

        cms_c = CountMinSketch(width=10, depth=3, seed=2)  # different seed -> different hash seeds
        with self.assertRaises(ValueError):
            cms_a.merge(cms_c)


def benchmark_heavy_hitters(num_items: int = 100_000, heavy_ratio: float = 0.01) -> Tuple[float, List[Tuple[Any, int]]]:
    """
    Simple benchmark: build a sketch, insert a stream, and query heavy hitters.
    Returns the elapsed time (seconds) and the list of detected heavy hitters.
    """
    import time

    cms = CountMinSketch.with_error(epsilon=0.001, delta=1e-5, seed=999)

    heavy_count = int(num_items * heavy_ratio)
    light_count = num_items - heavy_count

    heavy_item = "HEAVY"
    # Insert heavy item many times.
    for _ in range(heavy_count):
        cms.add(heavy_item)

    # Insert many distinct light items once each.
    for i in range(light_count):
        cms.add(f"light_{i}")

    threshold = heavy_count // 2
    start = time.perf_counter()
    hitters = cms.heavy_hitters(threshold)
    elapsed = time.perf_counter() - start
    return elapsed, hitters


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run a quick benchmark demonstration
    elapsed, hitters = benchmark_heavy_hitters()
    print(f"\nBenchmark: heavy‑hitter query took {elapsed:.6f}s")
    print("Detected heavy hitters:", hitters)
