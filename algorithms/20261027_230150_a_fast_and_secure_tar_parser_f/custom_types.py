from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class TarHeader:
    """Immutable representation of a POSIX ustar header block."""
    name: str               # File name (may include prefix)
    mode: int               # Permission bits
    uid: int                # Owner user ID
    gid: int                # Owner group ID
    size: int               # Size of file in bytes
    mtime: int              # Modification time (epoch)
    checksum: int           # Header checksum
    typeflag: str           # Entry type ('0' regular file, '5' directory, etc.)
    linkname: str           # Target name of link
    magic: str              # Must be "ustar"
    version: str            # Usually "00"
    uname: str              # Owner user name
    gname: str              # Owner group name
    devmajor: int           # Device major number
    devminor: int           # Device minor number
    prefix: str             # Prefix for long names

    @property
    def full_name(self) -> str:
        """Combine prefix and name according to the ustar spec."""
        if self.prefix:
            return f"{self.prefix}/{self.name}"
        return self.name


@dataclass(slots=True)
class TarEntry:
    """A single entry extracted from a tar archive."""
    header: TarHeader
    data: bytes             # Raw file contents; empty for directories and special entries
