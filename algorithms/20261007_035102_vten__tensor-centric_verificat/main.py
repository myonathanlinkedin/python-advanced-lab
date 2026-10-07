from custom_types import DType, Tensor, AddOp, MatMulOp
from engine import VerificationEngine


def _assert_tensor(t: Tensor, expected_shape: tuple[int, ...], expected_dtype: DType) -> None:
    assert t.shape == expected_shape, f"Expected shape {expected_shape}, got {t.shape}"
    assert t.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {t.dtype}"
    if t.data is not None:
        expected_len = 1
        for d in expected_shape:
            expected_len *= d
        assert len(t.data) == expected_len, f"Data length mismatch: expected {expected_len}, got {len(t.data)}"


def run_demo() -> None:
    # Create base tensors
    a = Tensor(shape=(2, 3), dtype=DType.FLOAT32)
    b = Tensor(shape=(2, 3), dtype=DType.FLOAT32)
    c = Tensor(shape=(3, 4), dtype=DType.FLOAT32)

    # Valid operations
    add_op = AddOp(a, b)          # result shape (2,3)
    matmul_op = MatMulOp(a, c)    # result shape (2,4)

    engine = VerificationEngine()
    engine.add_operation(add_op)
    engine.add_operation(matmul_op)

    # Verify graph consistency
    engine.verify()

    # Execute dummy forward pass
    results = engine.execute()
    add_result, matmul_result = results

    # Assertions on results
    _assert_tensor(add_result, (2, 3), DType.FLOAT32)
    _assert_tensor(matmul_result, (2, 4), DType.FLOAT32)

    # Edge‑case tests – expected failures
    # 1. Shape mismatch for AddOp
    try:
        AddOp(Tensor(shape=(2, 3), dtype=DType.INT32),
               Tensor(shape=(3, 2), dtype=DType.INT32))
        assert False, "AddOp should have raised ValueError for shape mismatch"
    except ValueError:
        pass

    # 2. Dtype mismatch for AddOp
    try:
        AddOp(Tensor(shape=(2, 3), dtype=DType.FLOAT32),
               Tensor(shape=(2, 3), dtype=DType.INT32))
        assert False, "AddOp should have raised ValueError for dtype mismatch"
    except ValueError:
        pass

    # 3. Non‑2D tensor for MatMulOp
    try:
        MatMulOp(Tensor(shape=(2, 3, 4), dtype=DType.FLOAT32),
                 Tensor(shape=(3, 4), dtype=DType.FLOAT32))
        assert False, "MatMulOp should have raised ValueError for non‑2D input"
    except ValueError:
        pass

    # 4. Inner dimension mismatch for MatMulOp
    try:
        MatMulOp(Tensor(shape=(2, 5), dtype=DType.FLOAT32),
                 Tensor(shape=(3, 4), dtype=DType.FLOAT32))
        assert False, "MatMulOp should have raised ValueError for inner dimension mismatch"
    except ValueError:
        pass

    # 5. Dtype mismatch for MatMulOp
    try:
        MatMulOp(Tensor(shape=(2, 3), dtype=DType.FLOAT32),
                 Tensor(shape=(3, 4), dtype=DType.INT32))
        assert False, "MatMulOp should have raised ValueError for dtype mismatch"
    except ValueError:
        pass

    print("All assertions passed. Demo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
