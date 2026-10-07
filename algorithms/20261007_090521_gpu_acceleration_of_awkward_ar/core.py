from __future__ import annotations
from typing import Any, Callable, List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


class AwkwardArray:
    """
    Minimal ragged array supporting element‑wise operations via a simulated
    GPU compute using a thread pool.
    """

    def __init__(self, data: Any) -> None:
        self._data = data
        self._shape = self._infer_shape(data)

    @property
    def data(self) -> Any:
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def _infer_shape(self, obj: Any) -> Tuple[int, ...]:
        """Recursively infer the shape of a ragged nested list."""
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            if not obj:  # empty list
                return (0,)
            first = obj[0]
            subshape = self._infer_shape(first)
            return (len(obj),) + subshape
        return ()

    def flatten(self) -> List[Any]:
        """Return a flat list of leaf values."""
        flat: List[Any] = []

        def _walk(o: Any) -> None:
            if isinstance(o, Sequence) and not isinstance(o, (str, bytes)):
                for item in o:
                    _walk(item)
            else:
                flat.append(o)

        _walk(self._data)
        return flat

    @classmethod
    def _unflatten(cls, flat: List[Any], shape: Tuple[int, ...]) -> Any:
        """Reconstruct nested structure from flat list using the provided shape."""
        if not shape:
            # scalar
            return flat.pop(0)

        length, *subshape = shape
        return [cls._unflatten(flat, tuple(subshape)) for _ in range(length)]

    def unflatten(self, flat: List[Any]) -> AwkwardArray:
        """Create a new AwkwardArray from a flat list using this instance's shape."""
        if len(flat) != self._num_leaves():
            raise ValueError("Flat list length does not match number of leaves.")
        # work on a copy to avoid mutating caller's list
        copy = list(flat)
        data = self._unflatten(copy, self._shape)
        return AwkwardArray(data)

    def _num_leaves(self) -> int:
        """Count leaf elements."""
        return len(self.flatten())

    def _check_compatible(self, other: AwkwardArray) -> None:
        """Ensure two arrays have identical ragged structure."""
        if self._shape != other._shape:
            raise ValueError(f"Incompatible shapes: {self._shape} vs {other._shape}")

    def gpu_compute(
        self,
        other: AwkwardArray,
        func: Callable[[Any, Any], Any],
        max_workers: int | None = None,
    ) -> AwkwardArray:
        """
        Simulate GPU‑accelerated element‑wise computation.

        The operation is applied to each pair of leaf values in parallel using a
        thread pool. The resulting flat list is reshaped back to the original ragged
        structure.
        """
        self._check_compatible(other)

        flat_a = self.flatten()
        flat_b = other.flatten()
        n = len(flat_a)

        # Determine worker count; cap to a reasonable number for the simulation.
        workers = max_workers or min(32, n) or 1

        results: List[Any] = [None] * n

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(func, flat_a[i], flat_b[i]): i for i in range(n)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return self.unflatten(results)

    def __repr__(self) -> str:
        return f"AwkwardArray(shape={self._shape}, data={self._data})"
