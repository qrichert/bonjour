#!/usr/bin/env python3
"""Prepare a minimal, deterministic holdout source from Meta Kaggle users."""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

DEFAULT_SAMPLE_SIZE = 2_000
DEFAULT_RNG_SEED = 0x5245414C
EXPECTED_HEADER = (
    "Id",
    "UserName",
    "DisplayName",
    "RegisterDate",
    "PerformanceTier",
    "Country",
    "LocationSharingOptOut",
    "ProgressionOptedOut",
)
OUTPUT_HEADER = ("display_name", "country_hint", "locale_hint")


class HashingReader(io.RawIOBase):
    """Binary reader that hashes bytes consumed through ``readinto``."""

    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        count = self.source.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


def parse_integer(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer (decimal or 0x-prefixed), got {value!r}"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uniformly sample Meta Kaggle user rows into the minimal source "
            "format accepted by the sealed holdout labeler."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Meta Kaggle Users.csv")
    parser.add_argument("output_csv", type=Path, help="New holdout source CSV")
    parser.add_argument(
        "provenance_json",
        type=Path,
        help="New aggregate provenance JSON",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"number of user rows to sample (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=parse_integer,
        default=DEFAULT_RNG_SEED,
        help=f"RNG seed in decimal or 0x notation (default: 0x{DEFAULT_RNG_SEED:08X})",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        type=Path,
        default=[],
        help=(
            "minimal holdout source CSV whose exact display-name values must "
            "be excluded; may be repeated"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_display_names(
    source_path: Path,
    sample_size: int,
    rng_seed: int,
    excluded_display_names: set[str] | None = None,
) -> tuple[list[str], dict[str, int | str]]:
    excluded_display_names = excluded_display_names or set()
    rng = random.Random(rng_seed)
    reservoir: list[str] = []
    source_rows = 0
    nonblank_rows = 0
    eligible_rows = 0
    excluded_blank_rows = 0
    excluded_exact_rows = 0

    with source_path.open("rb") as binary_source:
        hashing_source = HashingReader(binary_source)
        with open_text_reader(hashing_source) as text_source:
            reader = csv.reader(text_source)
            header = tuple(next(reader, ()))
            if header != EXPECTED_HEADER:
                raise ValueError(
                    "Meta Kaggle Users.csv header changed: expected "
                    f"{EXPECTED_HEADER!r}, got {header!r}"
                )

            for row in reader:
                source_rows += 1
                if len(row) != len(EXPECTED_HEADER):
                    raise ValueError(
                        f"{source_path}:{reader.line_num}: expected "
                        f"{len(EXPECTED_HEADER)} fields, got {len(row)}"
                    )
                display_name = row[2]
                if not display_name.strip():
                    excluded_blank_rows += 1
                    continue

                nonblank_rows += 1
                if display_name in excluded_display_names:
                    excluded_exact_rows += 1
                    continue

                eligible_rows += 1
                if len(reservoir) < sample_size:
                    reservoir.append(display_name)
                    continue

                replacement = rng.randrange(eligible_rows)
                if replacement < sample_size:
                    reservoir[replacement] = display_name

        source_sha256_before = hashing_source.digest.hexdigest()

    if eligible_rows < sample_size:
        raise ValueError(
            f"requested {sample_size:,} rows but source has only "
            f"{eligible_rows:,} nonblank display names"
        )

    return reservoir, {
        "source_rows": source_rows,
        "nonblank_rows_before_exact_exclusions": nonblank_rows,
        "eligible_nonblank_rows": eligible_rows,
        "excluded_blank_or_whitespace_rows": excluded_blank_rows,
        "excluded_exact_display_name_rows": excluded_exact_rows,
        "source_sha256_before": source_sha256_before,
    }


def open_text_reader(source: HashingReader):
    buffered = io.BufferedReader(source)
    return io.TextIOWrapper(buffered, encoding="utf-8", newline="")


def write_output(path: Path, display_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(OUTPUT_HEADER)
        writer.writerows((display_name, "", "") for display_name in display_names)
        destination.flush()
        os.fsync(destination.fileno())


def write_json(path: Path, value: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())


def temporary_path(destination: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def publish_pair(
    output_temporary: Path,
    output_path: Path,
    provenance_temporary: Path,
    provenance_path: Path,
) -> None:
    output_published = False
    provenance_published = False
    try:
        os.link(output_temporary, output_path)
        output_published = True
        os.link(provenance_temporary, provenance_path)
        provenance_published = True
    except BaseException:
        if output_published:
            output_path.unlink(missing_ok=True)
        if provenance_published:
            provenance_path.unlink(missing_ok=True)
        raise


def load_exclusion_sources(
    paths: list[Path],
) -> tuple[set[str], list[dict[str, object]]]:
    excluded: set[str] = set()
    provenance: list[dict[str, object]] = []
    for path in paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"not a file: {resolved}")
        with resolved.open(encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            header = tuple(next(reader, ()))
            if header != OUTPUT_HEADER:
                raise ValueError(
                    f"exclusion source {resolved} has unsupported header: "
                    f"expected {OUTPUT_HEADER!r}, got {header!r}"
                )
            rows = 0
            source_values: set[str] = set()
            for row in reader:
                rows += 1
                if len(row) != len(OUTPUT_HEADER):
                    raise ValueError(
                        f"{resolved}:{reader.line_num}: expected "
                        f"{len(OUTPUT_HEADER)} fields, got {len(row)}"
                    )
                display_name = row[0]
                if not display_name.strip():
                    raise ValueError(
                        f"{resolved}:{reader.line_num}: exclusion source contains "
                        "an empty display_name"
                    )
                source_values.add(display_name)
        excluded.update(source_values)
        provenance.append(
            {
                "file": resolved.name,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
                "rows": rows,
                "unique_display_names": len(source_values),
            }
        )
    return excluded, provenance


def prepare(
    source_path: Path,
    output_path: Path,
    provenance_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    rng_seed: int = DEFAULT_RNG_SEED,
    exclude_source_paths: list[Path] | None = None,
) -> dict[str, object]:
    source_path = source_path.resolve(strict=True)
    output_path = output_path.absolute()
    provenance_path = provenance_path.absolute()
    exclude_source_paths = exclude_source_paths or []
    resolved_exclusions = [path.resolve(strict=True) for path in exclude_source_paths]

    if not source_path.is_file():
        raise ValueError(f"not a file: {source_path}")
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    if rng_seed < 0:
        raise ValueError("RNG seed must be nonnegative")
    if not output_path.parent.is_dir():
        raise ValueError(f"output parent does not exist: {output_path.parent}")
    if not provenance_path.parent.is_dir():
        raise ValueError(f"provenance parent does not exist: {provenance_path.parent}")
    all_paths = [source_path, output_path, provenance_path, *resolved_exclusions]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("input, outputs, and exclusion sources must all differ")
    if os.path.lexists(output_path):
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    if os.path.lexists(provenance_path):
        raise FileExistsError(f"refusing to overwrite: {provenance_path}")

    excluded_display_names, exclusion_provenance = load_exclusion_sources(
        resolved_exclusions
    )
    display_names, statistics = sample_display_names(
        source_path,
        sample_size,
        rng_seed,
        excluded_display_names,
    )
    output_temporary = temporary_path(output_path)
    provenance_temporary = temporary_path(provenance_path)

    try:
        write_output(output_temporary, display_names)
        source_sha256_after = sha256_file(source_path)
        if source_sha256_after != statistics["source_sha256_before"]:
            raise RuntimeError("source checksum changed while preparing holdout")

        provenance: dict[str, object] = {
            "format_version": 1,
            "source_dataset": "kaggle/meta-kaggle",
            "source_file": source_path.name,
            "source_size_bytes": source_path.stat().st_size,
            "source_sha256_before": statistics["source_sha256_before"],
            "source_sha256_after": source_sha256_after,
            "source_rows": statistics["source_rows"],
            "nonblank_rows_before_exact_exclusions": statistics[
                "nonblank_rows_before_exact_exclusions"
            ],
            "eligible_nonblank_rows": statistics["eligible_nonblank_rows"],
            "excluded_blank_or_whitespace_rows": statistics[
                "excluded_blank_or_whitespace_rows"
            ],
            "excluded_exact_display_name_rows": statistics[
                "excluded_exact_display_name_rows"
            ],
            "excluded_unique_display_names": len(excluded_display_names),
            "exclusion_sources": exclusion_provenance,
            "sample_method": (
                "uniform reservoir sample over nonblank user rows after exact "
                "display-name exclusions"
            ),
            "sample_size": sample_size,
            "rng": "Python random.Random (MT19937)",
            "rng_seed_decimal": rng_seed,
            "rng_seed_hex": f"0x{rng_seed:X}",
            "country_hint_policy": "empty; Meta Kaggle Country is not ISO alpha-2",
            "locale_hint_policy": "empty; Meta Kaggle provides no locale",
            "output_file": output_path.name,
            "output_size_bytes": output_temporary.stat().st_size,
            "output_sha256": sha256_file(output_temporary),
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(provenance_temporary, provenance)
        publish_pair(
            output_temporary,
            output_path,
            provenance_temporary,
            provenance_path,
        )
        return provenance
    finally:
        output_temporary.unlink(missing_ok=True)
        provenance_temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    try:
        provenance = prepare(
            args.input_csv,
            args.output_csv,
            args.provenance_json,
            args.sample_size,
            args.seed,
            args.exclude_source,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from error

    print(f"Source rows: {provenance['source_rows']:,}")
    print(f"Eligible rows: {provenance['eligible_nonblank_rows']:,}")
    print(
        "Excluded blank/whitespace rows: "
        f"{provenance['excluded_blank_or_whitespace_rows']:,}"
    )
    print(
        "Excluded exact display-name rows: "
        f"{provenance['excluded_exact_display_name_rows']:,}"
    )
    print(
        "Excluded unique display names: "
        f"{provenance['excluded_unique_display_names']:,}"
    )
    print(f"Sample rows: {provenance['sample_size']:,}")
    print(f"RNG seed: {provenance['rng_seed_hex']}")
    print(f"Source SHA-256: {provenance['source_sha256_before']}")
    print(f"Output SHA-256: {provenance['output_sha256']}")


if __name__ == "__main__":
    main()
