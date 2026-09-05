import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import zstandard

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/package_name_data.py"
SPEC = importlib.util.spec_from_file_location("package_name_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


class PackageNameDataTests(unittest.TestCase):
    def test_tar_and_zstd_output_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            source = temporary / "source"
            source.mkdir()
            (source / "a").write_bytes(b"alpha")
            (source / "b").write_bytes(b"beta")
            first_tar = temporary / "first.tar"
            second_tar = temporary / "second.tar"
            SCRIPT.write_tar(first_tar, source, ["a", "b"])
            SCRIPT.write_tar(second_tar, source, ["a", "b"])
            self.assertEqual(first_tar.read_bytes(), second_tar.read_bytes())

            compressor = zstandard.ZstdCompressor(
                level=19,
                threads=0,
                write_content_size=True,
                write_checksum=True,
            )
            first = compressor.compress(first_tar.read_bytes())
            second = compressor.compress(second_tar.read_bytes())
            self.assertEqual(first, second)
            self.assertEqual(
                hashlib.sha256(first).digest(), hashlib.sha256(second).digest()
            )

            decompressed = zstandard.ZstdDecompressor().decompress(first)
            with tarfile.open(fileobj=io.BytesIO(decompressed)) as archive:
                members = archive.getmembers()
            self.assertEqual(
                [member.name for member in members],
                [
                    "bonjour-name-data-v1",
                    "bonjour-name-data-v1/a",
                    "bonjour-name-data-v1/b",
                ],
            )
            self.assertTrue(all(member.mtime == 0 for member in members))
            self.assertTrue(
                all(member.uid == 0 and member.gid == 0 for member in members)
            )


if __name__ == "__main__":
    unittest.main()
