#!/usr/bin/env python3
"""Extract exact first-name counts into mirrored country CSV files."""

import argparse
import csv
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop surnames and count exact first-name/gender tuples in each "
            "country CSV."
        )
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing CC.csv files."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="New directory in which to mirror the aggregated CC.csv files.",
    )
    parser.add_argument(
        "combined_output",
        type=Path,
        help="New CSV file containing the aggregated rows from every country.",
    )
    return parser.parse_args()


def extract_country(source_path: Path, destination_path: Path, combined_writer) -> None:
    country = source_path.stem
    counts: Counter[tuple[str, str]] = Counter()
    source_rows = 0
    empty_names = 0

    with source_path.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)

        for row in reader:
            source_rows += 1

            if len(row) != 4:
                raise ValueError(
                    f"{source_path}:{reader.line_num}: "
                    f"expected 4 fields, got {len(row)}"
                )

            first_name, _last_name, gender, row_country = row

            if row_country != country:
                raise ValueError(
                    f"{source_path}:{reader.line_num}: country {row_country!r} "
                    f"does not match filename country {country!r}"
                )

            if gender not in {"", "F", "M"}:
                raise ValueError(
                    f"{source_path}:{reader.line_num}: unexpected gender {gender!r}"
                )

            if not first_name:
                empty_names += 1
                continue

            # Deliberately preserve spelling, whitespace, accents, and compound
            # separators for later normalization decisions.
            counts[(first_name, gender)] += 1

    with destination_path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(("name", "country", "gender", "count"))

        for (first_name, gender), count in counts.items():
            row = (first_name, country, gender, count)
            writer.writerow(row)
            combined_writer.writerow(row)

        destination.flush()
        os.fsync(destination.fileno())

    print(
        f"{source_path.name}: {source_rows:,} rows, "
        f"{len(counts):,} unique tuples, {empty_names:,} empty names skipped",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    combined_output = args.combined_output.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    if not output_dir.parent.is_dir():
        raise SystemExit(f"Output parent directory does not exist: {output_dir.parent}")

    if not combined_output.parent.is_dir():
        raise SystemExit(
            f"Combined output parent directory does not exist: {combined_output.parent}"
        )

    if combined_output == output_dir:
        raise SystemExit("Output directory and combined output must be different paths")

    if os.path.lexists(output_dir):
        raise SystemExit(f"Refusing to overwrite: {output_dir}")

    if os.path.lexists(combined_output):
        raise SystemExit(f"Refusing to overwrite: {combined_output}")

    source_paths = sorted(input_dir.glob("*.csv"))
    if not source_paths:
        raise SystemExit(f"No CSV files found in: {input_dir}")

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        # The file must close before its path is linked into place.
        combined_temporary = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{combined_output.name}.",
            suffix=".tmp",
            dir=combined_output.parent,
            delete=False,
        )
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    combined_temporary_path = Path(combined_temporary.name)
    combined_published = False

    try:
        with combined_temporary as combined_destination:
            combined_writer = csv.writer(combined_destination, lineterminator="\n")
            combined_writer.writerow(("name", "country", "gender", "count"))

            for source_path in source_paths:
                extract_country(
                    source_path,
                    temporary_dir / source_path.name,
                    combined_writer,
                )

            combined_destination.flush()
            os.fsync(combined_destination.fileno())

        # Publish only after every country completed successfully. Both
        # temporary outputs are siblings of their destinations.
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Refusing to overwrite: {output_dir}")
        if os.path.lexists(combined_output):
            raise FileExistsError(f"Refusing to overwrite: {combined_output}")

        os.link(combined_temporary_path, combined_output)
        combined_published = True
        temporary_dir.rename(output_dir)
    except BaseException:
        if combined_published:
            combined_output.unlink(missing_ok=True)
        combined_temporary_path.unlink(missing_ok=True)
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    # The combined output is now a hard link to the completed temporary file.
    combined_temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
