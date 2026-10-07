from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Any


class DType(Enum):
    FLOAT32 = auto()
    INT32 = auto()


@dataclass(frozen=True)
class Tensor:
    shape: Tuple[int, ...]
    dtype: DType
    data: Tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if self.data is not None:
            expected = 1
            for dim in self.shape:
                expected *= dim
            if len(self.data) != expected:
                raise ValueError(f"Data length {len(self.data)} does not match shape {self.shape}")


class Operation(ABC):
    """Base class for all tensor operations."""

    name: str
    inputs: Tuple[Tensor, ...]
    output: Tensor

    @abstractmethod
    def infer_output(self) -> Tensor:
        """Return a Tensor describing the output shape and dtype."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(inputs={self.inputs}, output={self.output})"


class AddOp(Operation):
    """Element‑wise addition of two tensors with identical shape and dtype."""

    def __init__(self, a: Tensor, b: Tensor) -> None:
        self.name = "add"
        self.inputs = (a, b)
        self.output = self.infer_output()

    def infer_output(self) -> Tensor:
        a, b = self.inputs
        if a.shape != b.shape:
            raise ValueError("AddOp requires identical shapes")
        if a.dtype != b.dtype:
            raise ValueError("AddOp requires identical dtypes")
        return Tensor(shape=a.shape, dtype=a.dtype)


class MatMulOp(Operation):
    """Matrix multiplication for 2‑D tensors."""

    def __init__(self, a: Tensor, b: Tensor) -> None:
        self.name = "matmul"
        self.inputs = (a, b)
        self.output = self.infer_output()

    def infer_output(self) -> Tensor:
        a, b = self.inputs
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("MatMulOp only supports 2‑D tensors")
        if a.shape[1] != b.shape[0]:
            raise ValueError("Inner dimensions must match for matmul")
        if a.dtype != b.dtype:
            raise ValueError("MatMulOp requires identical dtypes")
        out_shape = (a.shape[0], b.shape[1])
        return Tensor(shape=out_shape, dtype=a.dtype)
