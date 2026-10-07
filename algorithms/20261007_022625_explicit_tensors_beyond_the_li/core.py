from __future__ import annotations
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union

Number = Union[int, float, complex]

def _product(seq: Sequence[int]) -> int:
    prod = 1
    for n in seq:
        prod *= n
    return prod

def _validate_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    if not shape:
        raise ValueError("Shape must be a non‑empty sequence of positive integers")
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Invalid dimension size: {dim}")
    return tuple(shape)

def _nested_iter(data: Any) -> Iterable[Number]:
    if isinstance(data, (list, tuple)):
        for sub in data:
            yield from _nested_iter(sub)
    else:
        yield data

def _nested_shape(data: Any) -> Tuple[int, ...]:
    if not isinstance(data, (list, tuple)):
        return ()
    if not data:
        raise ValueError("Cannot infer shape from empty list")
    first_dim = len(data)
    sub_shape = _nested_shape(data[0])
    for sub in data[1:]:
        if _nested_shape(sub) != sub_shape:
            raise ValueError("Inconsistent sub‑shapes in nested data")
    return (first_dim, ) + sub_shape

def _build_nested(shape: Tuple[int, ...], fill: Number) -> List[Any]:
    if not shape:
        return fill
    return [_build_nested(shape[1:], fill) for _ in range(shape[0])]

def _get_nested(data: Any, idx: Tuple[int, ...]) -> Number:
    for i in idx:
        data = data[i]
    return data

def _set_nested(data: Any, idx: Tuple[int, ...], value: Number) -> None:
    for i in idx[:-1]:
        data = data[i]
    data[idx[-1]] = value

class Tensor:
    """
    Simple explicit‑tensor implementation that stores values as nested Python lists.
    Supports indexing, reshaping, transposition and element‑wise arithmetic.
    """
    __slots__ = ("_data", "_shape")

    def __init__(self,
                 data: Union[Number, Sequence[Any], None] = None,
                 shape: Sequence[int] | None = None,
                 fill: Number = 0) -> None:
        if data is None and shape is None:
            raise ValueError("Either data or shape must be provided")
        if shape is not None:
            shape = _validate_shape(shape)
            if data is None:
                self._data = _build_nested(shape, fill)
                self._shape = shape
                return
            # data supplied together with shape – verify compatibility
            inferred = _nested_shape(data)
            if inferred != tuple(shape):
                raise ValueError(f"Provided shape {shape} does not match data shape {inferred}")
            self._data = data  # type: ignore[arg-type]
            self._shape = shape
        else:
            # shape derived from data
            self._shape = _nested_shape(data)
            self._data = data  # type: ignore[arg-type]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return _product(self._shape)

    def __repr__(self) -> str:
        return f"Tensor(shape={self._shape}, data={self._data})"

    # --------------------------------------------------------------------- #
    # Indexing utilities
    # --------------------------------------------------------------------- #
    def _normalize_index(self, idx: Any) -> Tuple[Tuple[int, ...], bool]:
        """
        Returns a tuple of integer indices and a flag indicating whether the
        result should be a sub‑tensor (True) or a scalar (False).
        """
        if isinstance(idx, tuple):
            indices = idx
        else:
            indices = (idx,)

        if any(isinstance(i, slice) for i in indices):
            # Slice handling – produce a sub‑tensor
            return self._apply_slice(indices), True
        if len(indices) > self.ndim:
            raise IndexError("Too many indices for tensor")
        # Pad missing dimensions with full slices (treated as sub‑tensor)
        full_idx = tuple(indices) + (slice(None),) * (self.ndim - len(indices))
        if any(isinstance(i, slice) for i in full_idx):
            return self._apply_slice(full_idx), True
        # All integers – scalar access
        int_idx = tuple(int(i) for i in full_idx)
        return int_idx, False

    def _apply_slice(self, slices: Tuple[Any, ...]) -> Tuple[int, ...]:
        """
        Recursively apply slices to the nested data and return the resulting
        nested list as a new Tensor's data.
        """
        def recurse(sub_data: Any, dim: int) -> Any:
            if dim == len(slices):
                return sub_data
            sl = slices[dim]
            if isinstance(sl, slice):
                return [recurse(sub_data[i], dim + 1) for i in range(*sl.indices(len(sub_data)))]
            else:
                return recurse(sub_data[sl], dim + 1)
        sliced = recurse(self._data, 0)
        return sliced  # type: ignore[return-value]

    def __getitem__(self, idx: Any) -> Any:
        norm_idx, is_sub = self._normalize_index(idx)
        if is_sub:
            # norm_idx is actually a nested list after slicing
            sub_shape = self._infer_shape_from_data(norm_idx)
            return Tensor(data=norm_idx, shape=sub_shape)
        return _get_nested(self._data, norm_idx)  # type: ignore[arg-type]

    def __setitem__(self, idx: Any, value: Number) -> None:
        norm_idx, is_sub = self._normalize_index(idx)
        if is_sub:
            raise TypeError("Cannot assign to a sliced sub‑tensor directly")
        _set_nested(self._data, norm_idx, value)  # type: ignore[arg-type]

    @staticmethod
    def _infer_shape_from_data(data: Any) -> Tuple[int, ...]:
        if not isinstance(data, (list, tuple)):
            return ()
        return (len(data),) + Tensor._infer_shape_from_data(data[0])

    # --------------------------------------------------------------------- #
    # Transformations
    # --------------------------------------------------------------------- #
    def reshape(self, new_shape: Sequence[int]) -> Tensor:
        new_shape = _validate_shape(new_shape)
        if _product(new_shape) != self.size:
            raise ValueError("Total size must remain unchanged for reshape")
        flat = list(_nested_iter(self._data))
        def build(shape: Tuple[int, ...], iterator: iter) -> List[Any]:
            if not shape:
                return next(iterator)
            return [build(shape[1:], iterator) for _ in range(shape[0])]
        new_data = build(tuple(new_shape), iter(flat))
        return Tensor(data=new_data, shape=new_shape)

    def transpose(self, axes: Sequence[int] | None = None) -> Tensor:
        if axes is None:
            axes = reversed(range(self.ndim))
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError("Axes must be a permutation of dimensions")
        # Generate all index tuples in original order, then permute
        import itertools
        new_shape = tuple(self._shape[i] for i in axes)
        flat = {}
        for idx in itertools.product(*[range(d) for d in self._shape]):
            val = _get_nested(self._data, idx)
            permuted = tuple(idx[i] for i in axes)
            flat[permuted] = val
        # Rebuild nested list from flat dict
        def build(shape: Tuple[int, ...], prefix: Tuple[int, ...]) -> List[Any]:
            if not shape:
                return flat[prefix]
            return [build(shape[1:], prefix + (i,)) for i in range(shape[0])]
        new_data = build(new_shape, ())
        return Tensor(data=new_data, shape=new_shape)

    # --------------------------------------------------------------------- #
    # Element‑wise operations
    # --------------------------------------------------------------------- #
    def _binary_op(self, other: Any, op: Callable[[Number, Number], Number]) -> Tensor:
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Shapes must match for element‑wise operation")
            def recur(a: Any, b: Any) -> Any:
                if isinstance(a, list):
                    return [recur(ai, bi) for ai, bi in zip(a, b)]
                return op(a, b)
            new_data = recur(self._data, other._data)
            return Tensor(data=new_data, shape=self.shape)
        else:
            # Broadcast scalar
            def recur(a: Any) -> Any:
                if isinstance(a, list):
                    return [recur(ai) for ai in a]
                return op(a, other)
            new_data = recur(self._data)
            return Tensor(data=new_data, shape=self.shape)

    def __add__(self, other: Any) -> Tensor:  # type: ignore[override]
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: Any) -> Tensor:  # type: ignore[override]
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(self, other: Any) -> Tensor:  # type: ignore[override]
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> Tensor:  # type: ignore[override]
        return self._binary_op(other, lambda a, b: a / b)

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def tolist(self) -> List[Any]:
        """Return a deep copy of the nested list representation."""
        import copy
        return copy.deepcopy(self._data)

    def map(self, func: Callable[[Number], Number]) -> Tensor:
        """Apply func element‑wise and return a new Tensor."""
        def recur(x: Any) -> Any:
            if isinstance(x, list):
                return [recur(v) for v in x]
            return func(x)
        new_data = recur(self._data)
        return Tensor(data=new_data, shape=self.shape)
