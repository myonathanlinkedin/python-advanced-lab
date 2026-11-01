from __future__ import annotations

from typing import List, Tuple
from .types import QRMatrix

# ---------------------------------------------------------------------------
# Alphanumeric character set as defined by the QR specification.
# ---------------------------------------------------------------------------
_ALPHANUMERIC_TABLE: List[str] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', ' ', '$', '%', '*',
    '+', '-', '.', '/', ':'
]
_ALPHANUMERIC_MAP = {c: i for i, c in enumerate(_ALPHANUMERIC_TABLE)}

# ---------------------------------------------------------------------------
# Core constants for a Version‑1 QR Code (21×21 modules) with EC‑Level L.
# ---------------------------------------------------------------------------
_VERSION = 1
_SIZE = 21                     # side length in modules
_DATA_CODEWORDS = 19           # number of data codewords for V1‑L
_TOTAL_BITS = _DATA_CODEWORDS * 8

# ---------------------------------------------------------------------------
# Helper functions for bit manipulation.
# ---------------------------------------------------------------------------
def _int_to_bits(value: int, length: int) -> List[int]:
    """Return the `length`‑bit big‑endian representation of `value`."""
    return [(value >> (length - 1 - i)) & 1 for i in range(length)]

def _bits_to_int(bits: List[int]) -> int:
    """Convert a big‑endian list of bits to an integer."""
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v

# ---------------------------------------------------------------------------
# Matrix construction utilities.
# ---------------------------------------------------------------------------
def _empty_matrix() -> QRMatrix:
    """Create a SIZE×SIZE matrix initialised to False (white)."""
    return [[False for _ in range(_SIZE)] for _ in range(_SIZE)]

def _mark_reserved(reserved: List[List[bool]], r: int, c: int) -> None:
    """Mark a cell as part of the function pattern."""
    reserved[r][c] = True

def _add_finder_pattern(matrix: QRMatrix, reserved: List[List[bool]], top: int, left: int) -> None:
    """Place a 7×7 finder pattern with its separator."""
    # 7×7 pattern (black = True, white = False)
    pattern = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    for dr in range(7):
        for dc in range(7):
            matrix[top + dr][left + dc] = bool(pattern[dr][dc])
            _mark_reserved(reserved, top + dr, left + dc)

    # Separator (one white module around the pattern)
    for dr in range(-1, 8):
        for dc in [-1, 7]:
            r, c = top + dr, left + dc
            if 0 <= r < _SIZE and 0 <= c < _SIZE:
                matrix[r][c] = False
                _mark_reserved(reserved, r, c)
    for dr in [-1, 7]:
        for dc in range(-1, 8):
            r, c = top + dr, left + dc
            if 0 <= r < _SIZE and 0 <= c < _SIZE:
                matrix[r][c] = False
                _mark_reserved(reserved, r, c)

def _add_timing_patterns(matrix: QRMatrix, reserved: List[List[bool]]) -> None:
    """Add the horizontal and vertical timing patterns."""
    for i in range(8, _SIZE - 8):
        val = (i % 2 == 0)
        # Horizontal timing (row 6)
        matrix[6][i] = val
        _mark_reserved(reserved, 6, i)
        # Vertical timing (column 6)
        matrix[i][6] = val
        _mark_reserved(reserved, i, 6)

def _add_dark_module(matrix: QRMatrix, reserved: List[List[bool]]) -> None:
    """Add the single dark module required by the spec (for V1 it is at (13,8))."""
    r, c = 13, 8
    matrix[r][c] = True
    _mark_reserved(reserved, r, c)

def _is_reserved(reserved: List[List[bool]], r: int, c: int) -> bool:
    return reserved[r][c]

# ---------------------------------------------------------------------------
# Data placement according to the QR specification (Version‑1, mask‑0).
# ---------------------------------------------------------------------------
def _place_data_bits(matrix: QRMatrix, reserved: List[List[bool]], data_bits: List[int]) -> None:
    """Place the data bits into the matrix following the standard zig‑zag pattern."""
    size = _SIZE
    bit_idx = 0
    col = size - 1

    while col > 0:
        # Skip the vertical timing pattern column.
        if col == 6:
            col -= 1

        for row_offset in range(size):
            # Determine the actual row depending on the direction.
            row = size - 1 - row_offset if ((size - col) // 2) % 2 == 0 else row_offset

            for dc in (0, -1):
                c = col + dc
                if _is_reserved(reserved, row, c):
                    continue
                if bit_idx < len(data_bits):
                    matrix[row][c] = bool(data_bits[bit_idx])
                    bit_idx += 1
                else:
                    matrix[row][c] = False  # pad with 0 if we run out (should not happen)
        col -= 2

# ---------------------------------------------------------------------------
# Encoding utilities (used only for the internal test harness).
# ---------------------------------------------------------------------------
def _encode_alphanumeric(data: str) -> List[int]:
    """Encode a string in alphanumeric mode and return a list of bits (0/1)."""
    bits: List[int] = []

    # Mode indicator (0010)
    bits.extend(_int_to_bits(0b0010, 4))

    # Character count (9 bits for version 1)
    bits.extend(_int_to_bits(len(data), 9))

    # Encode characters in pairs.
    i = 0
    while i + 1 < len(data):
        v = 45 * _ALPHANUMERIC_MAP[data[i]] + _ALPHANUMERIC_MAP[data[i + 1]]
        bits.extend(_int_to_bits(v, 11))
        i += 2
    if i < len(data):
        # Single trailing character (6 bits)
        v = _ALPHANUMERIC_MAP[data[i]]
        bits.extend(_int_to_bits(v, 6))

    # Terminator (up to 4 zero bits)
    terminator_len = min(4, _TOTAL_BITS - len(bits))
    bits.extend([0] * terminator_len)

    # Pad to a whole number of bytes.
    while len(bits) % 8 != 0:
        bits.append(0)

    # Pad with 0xEC, 0x11 alternately until we reach the required number of data codewords.
    pad_bytes = [0xEC, 0x11]
    byte_idx = 0
    while len(bits) < _TOTAL_BITS:
        bits.extend(_int_to_bits(pad_bytes[byte_idx % 2], 8))
        byte_idx += 1

    return bits

def encode_to_matrix(data: str) -> QRMatrix:
    """
    Public helper used by the test suite: encode `data` (alphanumeric only)
    into a Version‑1, EC‑L QR matrix using mask pattern 0.
    """
    if any(ch not in _ALPHANUMERIC_MAP for ch in data):
        raise ValueError("Only alphanumeric characters are supported in this minimal encoder.")

    matrix = _empty_matrix()
    reserved = [[False for _ in range(_SIZE)] for _ in range(_SIZE)]

    # Function patterns
    _add_finder_pattern(matrix, reserved, 0, 0)               # top‑left
    _add_finder_pattern(matrix, reserved, 0, _SIZE - 7)       # top‑right
    _add_finder_pattern(matrix, reserved, _SIZE - 7, 0)       # bottom‑left
    _add_timing_patterns(matrix, reserved)
    _add_dark_module(matrix, reserved)

    # Data bits
    data_bits = _encode_alphanumeric(data)
    _place_data_bits(matrix, reserved, data_bits)

    # No mask is applied (mask pattern 0 == identity)
    return matrix

# ---------------------------------------------------------------------------
# Decoding implementation (the core of the requested algorithm).
# ---------------------------------------------------------------------------
def _read_data_bits(matrix: QRMatrix, reserved: List[List[bool]]) -> List[int]:
    """Read the data bits from the matrix in the same order used for placement."""
    size = _SIZE
    bits: List[int] = []
    col = size - 1

    while col > 0:
        if col == 6:
            col -= 1
        for row_offset in range(size):
            row = size - 1 - row_offset if ((size - col) // 2) % 2 == 0 else row_offset
            for dc in (0, -1):
                c = col + dc
                if _is_reserved(reserved, row, c):
                    continue
                bits.append(1 if matrix[row][c] else 0)
        col -= 2
    return bits

def _decode_alphanumeric(bits: List[int]) -> str:
    """Decode a bit stream that is known to be alphanumeric mode."""
    # Mode (first 4 bits) – we assume it is 0010.
    mode = _bits_to_int(bits[:4])
    if mode != 0b0010:
        raise ValueError(f"Unsupported mode {mode:04b}")

    # Character count (next 9 bits for version 1)
    char_count = _bits_to_int(bits[4:13])
    pos = 13
    chars: List[str] = []

    # Decode pairs of characters.
    while len(chars) + 2 <= char_count:
        val = _bits_to_int(bits[pos:pos + 11])
        pos += 11
        first = val // 45
        second = val % 45
        chars.append(_ALPHANUMERIC_TABLE[first])
        chars.append(_ALPHANUMERIC_TABLE[second])

    # If an odd number of characters remain, decode the final 6‑bit value.
    if len(chars) < char_count:
        val = _bits_to_int(bits[pos:pos + 6])
        chars.append(_ALPHANUMERIC_TABLE[val])
        pos += 6

    return ''.join(chars)

def decode_qr(matrix: QRMatrix) -> str:
    """
    Decode a Version‑1, EC‑L QR matrix that uses alphanumeric mode and mask pattern 0.
    The function validates the matrix size, extracts the data bits, and returns the
    decoded string.
    """
    if len(matrix) != _SIZE or any(len(row) != _SIZE for row in matrix):
        raise ValueError(f"Matrix must be {_SIZE}×{_SIZE}")

    # Re‑create the reserved‑module map (identical to the encoder's construction).
    reserved = [[False for _ in range(_SIZE)] for _ in range(_SIZE)]
    _add_finder_pattern(matrix, reserved, 0, 0)               # top‑left
    _add_finder_pattern(matrix, reserved, 0, _SIZE - 7)       # top‑right
    _add_finder_pattern(matrix, reserved, _SIZE - 7, 0)       # bottom‑left
    _add_timing_patterns(matrix, reserved)
    _add_dark_module(matrix, reserved)

    # No mask is applied (mask pattern 0), so we can read bits directly.
    raw_bits = _read_data_bits(matrix, reserved)

    # Trim any trailing padding bits beyond the last full byte (the encoder pads to a full byte).
    # The decoder works on the full bit stream; the alphanumeric parser will stop after
    # reading the declared character count.
    return _decode_alphanumeric(raw_bits)

# =============================================================================
# End of engine.py
# =============================================================================
