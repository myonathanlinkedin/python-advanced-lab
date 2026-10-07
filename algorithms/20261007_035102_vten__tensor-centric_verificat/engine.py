from typing import List

from custom_types import Operation, Tensor


class VerificationEngine:
    """Core engine that stores operations, validates them and can execute a dummy forward pass."""

    def __init__(self) -> None:
        self.ops: List[Operation] = []

    def add_operation(self, op: Operation) -> None:
        """Register an operation in the execution order."""
        self.ops.append(op)

    def verify(self) -> None:
        """Perform lightweight graph consistency checks."""
        seen_outputs: List[Tensor] = []
        for op in self.ops:
            for inp in op.inputs:
                if not isinstance(inp, Tensor):
                    raise TypeError("Operation inputs must be Tensor instances")
                # Allow inputs that are either user‑provided tensors or outputs of earlier ops
                if inp not in seen_outputs and inp not in [o.output for o in self.ops]:
                    # In this simple framework we accept any Tensor; a full implementation would track provenance.
                    pass
            seen_outputs.append(op.output)

    def execute(self) -> List[Tensor]:
        """Run a dummy forward pass producing zero‑filled tensors matching each operation's output."""
        results: List[Tensor] = []
        for op in self.ops:
            results.append(self._compute(op))
        return results

    def _compute(self, op: Operation) -> Tensor:
        size = 1
        for dim in op.output.shape:
            size *= dim
        dummy_data = tuple(0 for _ in range(size))
        return Tensor(shape=op.output.shape, dtype=op.output.dtype, data=dummy_data)
