import struct
from dataclasses import dataclass
from typing import List, BinaryIO

@dataclass
class TarEntry:
    name: str
    mode: int
    uid: int
    gid: int
    size: int
    mtime: int
    typeflag: str
    data: bytes

class TarParser:
    BLOCK_SIZE = 512

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0
        self.entries: List[TarEntry] = []

    def _read_block(self) -> bytes:
        block = self._data[self._pos:self._pos + self.BLOCK_SIZE]
        self._pos += self.BLOCK_SIZE
        return block

    def _parse_header(self, block: bytes) -> TarEntry:
        if len(block) != self.BLOCK_SIZE:
            raise ValueError("Incomplete header block")
        # Extract fields
        name = block[0:100].rstrip(b'\0').decode('utf-8')
        mode = int(block[100:108].rstrip(b'\0').decode('utf-8'), 8)
        uid = int(block[108:116].rstrip(b'\0').decode('utf-8'), 8)
        gid = int(block[116:124].rstrip(b'\0').decode('utf-8'), 8)
        size = int(block[124:136].rstrip(b'\0').decode('utf-8'), 8)
        mtime = int(block[136:148].rstrip(b'\0').decode('utf-8'), 8)
        chksum = int(block[148:156].rstrip(b'\0').decode('utf-8'), 8)
        typeflag = block[156:157].decode('utf-8')
        # Verify checksum
        chk_block = block[:148] + b' ' * 8 + block[156:]
        if sum(chk_block) != chksum:
            raise ValueError(f"Checksum mismatch for {name}")
        # Security: reject unsafe paths
        if name.startswith('/') or '..' in name.split('/'):
            raise ValueError(f"Unsafe path: {name}")
        # Read file data
        data_start = self._pos
        data_end = data_start + size
        if data_end > len(self._data):
            raise ValueError(f"Truncated data for {name}")
        data = self._data[data_start:data_end]
        # Advance to next block boundary
        padding = (self.BLOCK_SIZE - (size % self.BLOCK_SIZE)) % self.BLOCK_SIZE
        self._pos = data_end + padding
        return TarEntry(name, mode, uid, gid, size, mtime, typeflag, data)

    def parse(self) -> List[TarEntry]:
        while self._pos < len(self._data):
            block = self._read_block()
            if all(b == 0 for b in block):
                # Peek at next block to confirm end of archive
                if self._pos + self.BLOCK_SIZE > len(self._data):
                    break
                next_block = self._data[self._pos:self._pos + self.BLOCK_SIZE]
                if all(b == 0 for b in next_block):
                    break
                # Skip this single zero block
                self._pos += self.BLOCK_SIZE
                continue
            entry = self._parse_header(block)
            self.entries.append(entry)
        return self.entries

def parse_tar(data: bytes) -> List[TarEntry]:
    parser = TarParser(data)
    return parser.parse()
