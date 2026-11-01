from __future__ import annotations

from typing import List

from .engine import encode_to_matrix, decode_qr
from .types import QRMatrix

def _matrix_to_string(matrix: QRMatrix) -> str:
    """Utility to visualise a QR matrix as ASCII (█ for black, space for white)."""
    return '\n'.join(''.join('█' if cell else ' ' for cell in row) for row in matrix)

def _run_test_case(data: str) -> None:
    """Encode `data`, decode it back, and assert correctness."""
    matrix = encode_to_matrix(data)
    decoded = decode_qr(matrix)
    assert decoded == data, f"Round‑trip failed: expected {data!r}, got {decoded!r}"
    # Optional visual sanity check (comment out for silent test runs)
    # print(f"Data: {data}")
    # print(_matrix_to_string(matrix))
    # print(f"Decoded: {decoded}\n")

def main() -> None:
    """Execute a suite of deterministic unit tests for the QR encoder/decoder."""
    test_cases: List[str] = [
        "",                                 # empty string
        "A",                                # single character
        "AB",                               # two characters (one pair)
        "HELLO",                            # odd length
        "HELLO WORLD",                      # typical phrase (contains space)
        "1234567890",                       # numeric only (still alphanumeric)
        "QR CODE 2026",                     # mixed letters, digits, space
    ]

    for case in test_cases:
        _run_test_case(case)

    # Additional sanity check: ensure the matrix dimensions are correct.
    sample = encode_to_matrix("TEST")
    assert len(sample) == 21 and all(len(row) == 21 for row in sample), "Matrix size mismatch"

    print("All unit tests passed successfully.")

if __name__ == '__main__':
    main()

# =============================================================================
# End of main.py
# =============================================================================
