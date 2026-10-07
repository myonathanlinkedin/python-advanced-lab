import time
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    duration: float
    ops_per_sec: float


def _timeit(func: Callable, *args, **kwargs) -> float:
    """Measure execution time of ``func`` with given arguments."""
    start = time.perf_counter()
    func(*args, **kwargs)
    end = time.perf_counter()
    return end - start


def _noop() -> None:
    """A minimal function used for call‑overhead benchmarking."""
    pass


def benchmark_loop(iterations: int = 10_000_000) -> float:
    """Simple for‑loop that does nothing."""
    def _inner() -> None:
        for _ in range(iterations):
            pass
    return _timeit(_inner)


def benchmark_function_calls(iterations: int = 10_000_000) -> float:
    """Measure the overhead of calling a trivial function."""
    def _inner() -> None:
        for _ in range(iterations):
            _noop()
    return _timeit(_inner)


def benchmark_list_comprehension(iterations: int = 5_000_000) -> float:
    """Create a list via comprehension."""
    def _inner() -> None:
        _ = [i for i in range(iterations)]
    return _timeit(_inner)


def benchmark_dict_lookup(iterations: int = 5_000_000) -> float:
    """Lookup keys in a pre‑filled dictionary."""
    d = {i: i for i in range(iterations)}
    def _inner() -> None:
        for i in range(iterations):
            _ = d[i]
    return _timeit(_inner)


def benchmark_string_concatenation(iterations: int = 2_000_000) -> float:
    """Concatenate strings using ``+=`` (inefficient pattern)."""
    def _inner() -> None:
        s = ""
        for i in range(iterations):
            s += str(i)
    return _timeit(_inner)


def run_all_benchmarks(iterations: int = 10_000_000) -> List[BenchmarkResult]:
    """Execute the full suite and return structured results."""
    suite: List[Tuple[str, Callable[[int], float]]] = [
        ("loop", benchmark_loop),
        ("function_calls", benchmark_function_calls),
        ("list_comprehension", benchmark_list_comprehension),
        ("dict_lookup", benchmark_dict_lookup),
        ("string_concatenation", benchmark_string_concatenation),
    ]

    results: List[BenchmarkResult] = []
    for name, func in suite:
        # Adjust iteration count for heavier workloads
        it = iterations
        if name in {"list_comprehension", "dict_lookup", "string_concatenation"}:
            it = max(1, iterations // 2)
        duration = func(it)
        ops = it / duration if duration > 0 else float("inf")
        results.append(BenchmarkResult(name=name, duration=duration, ops_per_sec=ops))
    return results
