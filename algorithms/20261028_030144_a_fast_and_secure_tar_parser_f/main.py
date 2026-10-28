import unittest
import io
import tarfile
from core import parse_tar, TarEntry

class TestTarParser(unittest.TestCase):
    def _make_tar(self, files: dict) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w') as tar:
            for name, content in files.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        return buf.getvalue()

    def test_basic(self):
        files = {'a.txt': b'hello', 'b.bin': b'\x00\x01\x02'}
        data = self._make_tar(files)
        entries = parse_tar(data)
        self.assertEqual(len(entries), 2)
        for e in entries:
            self.assertIn(e.name, files)
            self.assertEqual(e.data, files[e.name])

    def test_empty(self):
        data = self._make_tar({})
        entries = parse_tar(data)
        self.assertEqual(len(entries), 0)

    def test_path_traversal(self):
        files = {'../evil.txt': b'bad'}
        data = self._make_tar(files)
        with self.assertRaises(ValueError):
            parse_tar(data)

    def test_checksum_error(self):
        files = {'ok.txt': b'good'}
        data = self._make_tar(files)
        data = bytearray(data)
        data[148] ^= 0xFF  # corrupt checksum byte
        with self.assertRaises(ValueError):
            parse_tar(bytes(data))

    def test_large_file(self):
        content = b'a' * 1025
        files = {'large.bin': content}
        data = self._make_tar(files)
        entries = parse_tar(data)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].size, 1025)
        self.assertEqual(entries[0].data, content)

if __name__ == '__main__':
    unittest.main()
