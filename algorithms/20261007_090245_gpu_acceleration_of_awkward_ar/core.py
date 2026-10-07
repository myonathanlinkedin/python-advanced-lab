from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed


Leaf = Union[int, float, str, bool, None]
Nested = Union[Leaf, List["Nested"]]


def _is_leaf(value: Any) -> bool:
    return not isinstance(value, list)


def _flatten(data: Nested) -> List[Leaf]:
    """Recursively flatten a nested list structure into a list of leaf values."""
    if _is_leaf(data):
        return [data]  # type: ignore[return-value]
    flat: List[Leaf] = []
    for item in data:  # type: ignore[arg-type]
        flat.extend(_flatten(item))
    return flat


def _rebuild(template: Nested, values_iter: Iterable[Leaf]) -> Nested:
    """Rebuild a nested structure using values taken from values_iter."""
    if _is_leaf(template):
        return next(values_iter)  # type: ignore[return-value]
    rebuilt: List[Nested] = []
    for sub in template:  # type: ignore[arg-type]
        rebuilt.append(_rebuild(sub, values_iter))
    return rebuilt


def _traverse_shape(data: Nested) -> Tuple[int, ...]:
    """Return a tuple representing the shape of a regular (non‑ragged) nested list.
    Ragged dimensions are represented by -1."""
    if _is_leaf(data):
        return ()
    lengths = [len(data)]  # type: ignore[arg-type]
    # Examine first element to infer deeper shape; if ragged, use -1.
    first = data[0]  # type: ignore[index]
    sub_shape = _traverse_shape(first)
    # Verify regularity; if any sub‑list differs, mark ragged.
    for sub in data:  # type: ignore[arg-type]
        if _traverse_shape(sub) != sub_shape:
            sub_shape = tuple(-1 for _ in sub_shape)
            break
    return tuple(lengths) + sub_shape


@dataclass(frozen=True)
class AwkwardArray:
    """A minimal, immutable representation of a possibly ragged nested array."""
    data: Nested

    def __post_init__(self) -> None:
        if not isinstance(self.data, (list, int, float, str, bool, type(None))):
            raise TypeError("AwkwardArray data must be a nested list or a leaf value.")

    def __repr__(self) -> str:
        return f"AwkwardArray({self.data!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AwkwardArray):
            return NotImplemented
        return self.data == other.data

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the array; ragged dimensions are -1."""
        return _traverse_shape(self.data)

    def flatten(self) -> List[Leaf]:
        """Return a flat list of all leaf elements."""
        return _flatten(self.data)

    def map(self, func: Callable[[Leaf], Leaf]) -> "AwkwardArray":
        """Apply func to each leaf, preserving the original nesting."""
        flat = self.flatten()
        transformed = [func(v) for v in flat]
        rebuilt = _rebuild(self.data, iter(transformed))
        return AwkwardArray(rebuilt)


def cuda_compute(
    func: Callable[[Leaf], Leaf],
    array: AwkwardArray,
    *,
    parallel: bool = False,
    max_workers: int | None = None,
) -> AwkwardArray:
    """
    Simulated GPU‑accelerated element‑wise computation.

    Parameters
    ----------
    func:
        Callable applied to each leaf element.
    array:
        Input AwkwardArray.
    parallel:
        If True, computation is performed using a thread pool to mimic parallel execution.
    max_workers:
        Maximum number of worker threads; defaults to the system default.

    Returns
    -------
    AwkwardArray
        New array with ``func`` applied to every leaf.
    """
    flat = array.flatten()
    if not parallel:
        transformed = [func(v) for v in flat]
    else:
        transformed = [None] * len(flat)  # type: ignore[assignment]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(func, v): i for i, v in enumerate(flat)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                transformed[idx] = future.result()
    rebuilt = _rebuild(array.data, iter(transformed))
    return AwkwardArray(rebuilt)
