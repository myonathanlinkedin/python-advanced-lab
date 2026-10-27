from __future__ import annotations

import math
import struct
from typing import Generator, List

from .types import TarHeader, TarEntry

# Constants
BLOCK_SIZE = 512
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB safety limit


def _octal_to_int(octal_bytes: bytes) -> int:
    """Convert a null‑terminated or space‑padded octal byte string to int."""
    s = octal_bytes.partition(b'\0')[0].strip()
    if not s:
        return 0
    try:
        return int(s, 8)
    except ValueError as exc:
        raise ValueError(f"Invalid octal value: {octal_bytes!r}") from exc


def _parse_header(block: bytes) -> TarHeader:
    """Parse a 512‑byte header block into a TarHeader dataclass."""
    if len(block) != BLOCK_SIZE:
        raise ValueError("Header block must be exactly 512 bytes")

    # Unpack according to the POSIX ustar layout
    fields = struct.unpack(
        "100s8s8s8s12s12s8s1s100s6s2s32s32s8s8s155s12x",
        block,
    )
    (
        name,
        mode,
        uid,
        gid,
        size,
        mtime,
        checksum,
        typeflag,
        linkname,
        magic,
        version,
        uname,
        gname,
        devmajor,
        devminor,
        prefix,
    ) = fields

    # Decode textual fields (they are ASCII, possibly null‑padded)
    def decode(b: bytes) -> str:
        return b.partition(b'\0')[0].decode("utf-8", errors="replace")

    header = TarHeader(
        name=decode(name),
        mode=_octal_to_int(mode),
        uid=_octal_to_int(uid),
        gid=_octal_to_int(gid),
        size=_octal_to_int(size),
        mtime=_octal_to_int(mtime),
        checksum=_octal_to_int(checksum),
        typeflag=decode(typeflag),
        linkname=decode(linkname),
        magic=decode(magic),
        version=decode(version),
        uname=decode(uname),
        gname=decode(gname),
        devmajor=_octal_to_int(devmajor),
        devminor=_octal_to_int(devminor),
        prefix=decode(prefix),
    )
    return header


def _verify_checksum(block: bytes, header: TarHeader) -> None:
    """Validate the checksum of a header block; raise if mismatched."""
    # The checksum field is treated as if it were filled with spaces (0x20)
    checksum_field_slice = slice(148, 156)  # bytes 148‑155 inclusive
    checksum_bytes = bytearray(block)
    checksum_bytes[checksum_field_slice] = b" " * 8
    computed = sum(checksum_bytes)
    if computed != header.checksum:
        raise ValueError(
            f"Checksum mismatch for {header.full_name}: stored {header.checksum}, computed {computed}"
        )


def _is_end_of_archive(block: bytes) -> bool:
    """Two consecutive zero blocks signal the end of a tar archive."""
    return all(b == 0 for b in block)


def _security_checks(header: TarHeader) -> None:
    """Enforce path‑traversal and size limits."""
    name = header.full_name
    if name.startswith("/") or ".." in name.split("/"):
        raise ValueError(f"Unsafe path detected in tar entry: {name!r}")

    if header.size > MAX_FILE_SIZE:
        raise ValueError(
            f"File size {header.size} exceeds maximum allowed {MAX_FILE_SIZE} bytes for entry {name!r}"
        )


def parse_tar_stream(data: bytes) -> List[TarEntry]:
    """
    Parse a complete tar archive given as a bytes object.

    Returns a list of TarEntry objects in the order they appear.
    Raises ValueError on any malformed or insecure entry.
    """
    entries: List[TarEntry] = []
    offset = 0
    data_len = len(data)

    while offset + BLOCK_SIZE <= data_len:
        block = data[offset : offset + BLOCK_SIZE]
        if _is_end_of_archive(block):
            # Verify the second zero block per spec
            if offset + 2 * BLOCK_SIZE <= data_len:
                next_block = data[offset + BLOCK_SIZE : offset + 2 * BLOCK_SIZE]
                if _is_end_of_archive(next_block):
                    break
            # If we encounter a single zero block without a following one, treat as EOF
            break

        header = _parse_header(block)
        _verify_checksum(block, header)
        _security_checks(header)

        # Compute how many 512‑byte blocks the file data occupies
        data_blocks = math.ceil(header.size / BLOCK_SIZE)
        data_start = offset + BLOCK_SIZE
        data_end = data_start + data_blocks * BLOCK_SIZE

        if data_end > data_len:
            raise ValueError(f"Unexpected end of archive while reading data for {header.full_name}")

        file_data = data[data_start : data_start + header.size] if header.size > 0 else b""
        entries.append(TarEntry(header=header, data=file_data))

        # Advance to the next header block
        offset = data_end

    return entries
