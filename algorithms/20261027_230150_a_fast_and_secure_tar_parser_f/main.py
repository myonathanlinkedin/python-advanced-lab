from __future__ import annotations

import io
import tarfile
import unittest

from .engine import parse_tar_stream, MAX_FILE_SIZE
from .types import TarEntry


class TestTarParser(unittest.TestCase):
    def _make_tar_bytes(self, files: dict[str, bytes]) -> bytes:
        """Utility to create an in‑memory tar archive from a mapping of name→content."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for name, content in files.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(content)
                tf.addfile(tarinfo=info, fileobj=io.BytesIO(content))
        return buf.getvalue()

    def test_basic_parsing(self):
        payload = {
            "hello.txt": b"Hello, world!",
            "folder/readme.md": b"# Readme\nThis is a test.",
        }
        tar_bytes = self._make_tar_bytes(payload)
        entries = parse_tar_stream(tar_bytes)

        # Verify we got exactly the two entries (directories are not stored explicitly)
        self.assertEqual(len(entries), 2)
        names = {e.header.full_name for e in entries}
        self.assertSetEqual(names, set(payload.keys()))

        # Verify content integrity
        for entry in entries:
            expected = payload[entry.header.full_name]
            self.assertEqual(entry.data, expected)

    def test_path_traversal_detection(self):
        # Create a tar entry with a dangerous path
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
            info = tarfile.TarInfo(name="../evil.txt")
            data = b"malicious"
            info.size = len(data)
            tf.addfile(tarinfo=info, fileobj=io.BytesIO(data))
        tar_bytes = buf.getvalue()

        with self.assertRaises(ValueError) as ctx:
            parse_tar_stream(tar_bytes)
        self.assertIn("Unsafe path", str(ctx.exception))

    def test_oversized_file_detection(self):
        # Build a file just above the allowed limit
        oversized_content = b"a" * (MAX_FILE_SIZE + 1)
        tar_bytes = self._make_tar_bytes({"big.bin": oversized_content})

        with self.assertRaises(ValueError) as ctx:
            parse_tar_stream(tar_bytes)
        self.assertIn("exceeds maximum allowed", str(ctx.exception))

    def test_invalid_checksum_detection(self):
        # Create a valid tar then corrupt a byte in the header checksum field
        tar_bytes = self._make_tar_bytes({"good.txt": b"ok"})
        corrupted = bytearray(tar_bytes)
        # Corrupt one byte inside the checksum field (bytes 148‑155)
        corrupted[150] ^= 0xFF
        with self.assertRaises(ValueError) as ctx:
            parse_tar_stream(bytes(corrupted))
        self.assertIn("Checksum mismatch", str(ctx.exception))

    def test_end_of_archive_handling(self):
        # Ensure parser stops correctly at the double zero block
        tar_bytes = self._make_tar_bytes({"a.txt": b"1"})
        # Append extra random data after the proper end markers
        extra = b"EXTRA_DATA"
        data = tar_bytes + extra
        entries = parse_tar_stream(data)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].header.full_name, "a.txt")


if __name__ == "__main__":
    # Run the unit tests with high verbosity for demonstration purposes
    unittest.main(verbosity=2)
