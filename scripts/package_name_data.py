#!/usr/bin/env python3
"""Validate and deterministically package bonjour-name-data-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import zstandard

ARCHIVE_NAME = "bonjour-name-data-v1.tar.zst"
ROOT_NAME = "bonjour-name-data-v1"
DOCUMENTS = ("manifest.json", "README.md", "NOTICE")
REPOSITORY_FILES_DIRECTORY = "files"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("dist"))
    arguments = parser.parse_args()
    archive, checksum = package(
        arguments.artifact_directory, arguments.output_directory
    )
    print(f"Archive: {archive}")
    print(f"Checksum: {checksum}")


def package(artifact_directory: Path, output_directory: Path) -> tuple[Path, Path]:
    entries, constituent_directory = validate_artifact(artifact_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    archive = output_directory / ARCHIVE_NAME
    checksum = output_directory / f"{ARCHIVE_NAME}.sha256"
    if archive.exists() or checksum.exists():
        raise FileExistsError("refusing to overwrite name-data package output")

    with tempfile.NamedTemporaryFile(dir=output_directory, suffix=".tar") as temporary:
        write_tar(
            Path(temporary.name),
            artifact_directory,
            entries,
            constituent_directory=constituent_directory,
        )
        tar_bytes = Path(temporary.name).read_bytes()
    compressor = zstandard.ZstdCompressor(
        level=19,
        threads=0,
        write_content_size=True,
        write_checksum=True,
    )
    archive.write_bytes(compressor.compress(tar_bytes))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="ascii", newline="\n")
    return archive, checksum


def validate_artifact(artifact_directory: Path) -> tuple[list[str], Path]:
    repository_manifest = Path(__file__).parents[1] / "data/name-v1/manifest.json"
    pinned_bytes = repository_manifest.read_bytes()
    manifest_path = artifact_directory / "manifest.json"
    require_regular_file(manifest_path)
    if manifest_path.read_bytes() != pinned_bytes:
        raise ValueError(
            "artifact manifest does not match the pinned production manifest"
        )
    manifest = json.loads(pinned_bytes)
    repository_files = artifact_directory / REPOSITORY_FILES_DIRECTORY
    constituent_directory = (
        repository_files if repository_files.is_dir() else artifact_directory
    )
    entries = [*DOCUMENTS, *(file["name"] for file in manifest["files"])]
    if entries[3:] != sorted(entries[3:], key=lambda value: value.encode("utf-8")):
        raise ValueError("manifest constituents are not bytewise sorted")

    validate_digest(artifact_directory / "README.md", manifest["readme_sha256"])
    validate_digest(artifact_directory / "NOTICE", manifest["notice_sha256"])
    for expected in manifest["files"]:
        path = constituent_directory / expected["name"]
        require_regular_file(path)
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{path} has the wrong byte length")
        validate_digest(path, expected["sha256"])
    for path in artifact_directory.iterdir():
        if path.is_symlink():
            raise ValueError(f"refusing extra symlink: {path}")
    return sorted(
        entries, key=lambda value: value.encode("utf-8")
    ), constituent_directory


def write_tar(
    destination: Path,
    source: Path,
    entries: list[str],
    *,
    constituent_directory: Path | None = None,
) -> None:
    constituent_directory = constituent_directory or source
    with (
        destination.open("wb") as output,
        tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        root = canonical_info(ROOT_NAME, is_directory=True)
        archive.addfile(root)
        for name in entries:
            directory = source if name in DOCUMENTS else constituent_directory
            path = directory / name
            require_regular_file(path)
            info = canonical_info(f"{ROOT_NAME}/{name}", is_directory=False)
            info.size = path.stat().st_size
            with path.open("rb") as content:
                archive.addfile(info, content)


def canonical_info(name: str, *, is_directory: bool) -> tarfile.TarInfo:
    info = tarfile.TarInfo(f"{name}/" if is_directory else name)
    info.type = tarfile.DIRTYPE if is_directory else tarfile.REGTYPE
    info.mode = 0o755 if is_directory else 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required direct regular file is missing: {path}")


def validate_digest(path: Path, expected: str) -> None:
    require_regular_file(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{path} has the wrong checksum")


if __name__ == "__main__":
    main()
