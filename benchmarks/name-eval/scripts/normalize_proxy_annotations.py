#!/usr/bin/env python3
"""Normalize classifier-blind proxy annotations without repairing labels."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CANONICAL_HEADER = (
    "id",
    "display_name",
    "country_hint",
    "locale_hint",
    "decision",
    "expected_greeting",
)
IDENTITY_HEADER = CANONICAL_HEADER[:4]
POLICY = (
    "canonical GREETING/GREET decision fields preferred when present; "
    "otherwise legacy first_name; NULL/SKIP preserved; exact original-text "
    "spans become GREETING; empty, unsupported, or non-exact values become "
    "SKIP; confidence and notes ignored; raw annotations unchanged"
)


@dataclass(frozen=True)
class AnnotationInput:
    annotator: str
    raw_path: Path
    canonical_path: Path


@dataclass(frozen=True)
class SourceIdentity:
    id: str
    display_name: str
    country_hint: str
    locale_hint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mechanically normalize independent proxy annotations while "
            "preserving raw inputs."
        )
    )
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("template_csv", type=Path)
    parser.add_argument("provenance_json", type=Path)
    parser.add_argument(
        "--annotation",
        action="append",
        nargs=3,
        required=True,
        metavar=("ANNOTATOR", "RAW_CSV", "CANONICAL_CSV"),
        help="annotator label, returned raw CSV, and new canonical CSV",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_template(path: Path) -> list[SourceIdentity]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CANONICAL_HEADER:
            raise ValueError("annotation template header does not match the exchange format")
        rows = [source_identity(row) for row in reader]
    validate_identities(rows, "annotation template")
    return rows


def load_source_rows(path: Path) -> list[tuple[str, str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CANONICAL_HEADER[1:4]:
            raise ValueError("holdout source header does not match the minimal format")
        rows = [
            (row["display_name"], row["country_hint"], row["locale_hint"])
            for row in reader
        ]
    return sorted(rows)


def validate_source(template: list[SourceIdentity], source_path: Path) -> None:
    source_rows = load_source_rows(source_path)
    template_rows = sorted(
        (row.display_name, row.country_hint, row.locale_hint) for row in template
    )
    if source_rows != template_rows:
        raise ValueError("annotation template does not match the supplied holdout source")


def source_identity(row: dict[str, str]) -> SourceIdentity:
    return SourceIdentity(*(row[field] for field in IDENTITY_HEADER))


def validate_identities(rows: list[SourceIdentity], description: str) -> None:
    ids = [row.id for row in rows]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{description} contains an empty or duplicate ID")


def normalize_annotations(
    source_path: Path,
    template_path: Path,
    annotations: list[AnnotationInput],
    provenance_path: Path,
) -> dict[str, object]:
    if not annotations:
        raise ValueError("at least one annotation is required")
    labels = [annotation.annotator for annotation in annotations]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("annotator labels must be nonempty and unique")

    destinations = [annotation.canonical_path for annotation in annotations]
    destinations.append(provenance_path)
    if len(destinations) != len(set(destinations)):
        raise ValueError("canonical and provenance output paths must differ")
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(destination)

    template = load_template(template_path)
    validate_source(template, source_path)
    template_by_id = {row.id: row for row in template}
    raw_hashes_before = {
        annotation.annotator: sha256_file(annotation.raw_path)
        for annotation in annotations
    }
    canonical_outputs: list[tuple[Path, bytes]] = []
    annotation_provenance: list[dict[str, object]] = []
    invalid_case_sets: list[set[str]] = []

    for annotation in sorted(annotations, key=lambda value: value.annotator):
        canonical_rows, metadata, invalid_ids = normalize_one(
            annotation.raw_path,
            template,
            template_by_id,
        )
        canonical_bytes = serialize_canonical(canonical_rows)
        canonical_outputs.append((annotation.canonical_path, canonical_bytes))
        invalid_case_sets.append(invalid_ids)
        annotation_provenance.append(
            {
                "annotator": annotation.annotator,
                "raw_file": annotation.raw_path.name,
                "raw_sha256": raw_hashes_before[annotation.annotator],
                "canonical_file": annotation.canonical_path.name,
                "canonical_sha256": sha256_bytes(canonical_bytes),
                **metadata,
            }
        )

    for annotation in annotations:
        if sha256_file(annotation.raw_path) != raw_hashes_before[annotation.annotator]:
            raise RuntimeError("raw annotation changed during normalization")

    invalid_union = set().union(*invalid_case_sets)
    provenance = {
        "format_version": 1,
        "policy": POLICY,
        "source_csv_sha256": sha256_file(source_path),
        "template_sha256": sha256_file(template_path),
        "unusable_or_non_exact_cases": len(invalid_union),
        "annotations": annotation_provenance,
    }
    provenance_bytes = serialize_provenance(provenance)
    publish_outputs([*canonical_outputs, (provenance_path, provenance_bytes)])
    return provenance


def normalize_one(
    raw_path: Path,
    template: list[SourceIdentity],
    template_by_id: dict[str, SourceIdentity],
) -> tuple[list[dict[str, str]], dict[str, object], set[str]]:
    with raw_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = tuple(reader.fieldnames or ())
        if len(fields) != len(set(fields)):
            raise ValueError(f"{raw_path} contains duplicate columns")
        missing = set(IDENTITY_HEADER).difference(fields)
        if missing:
            raise ValueError(f"{raw_path} is missing identity columns: {sorted(missing)}")
        if "decision" not in fields and "first_name" not in fields:
            raise ValueError(f"{raw_path} has no supported annotation fields")
        raw_rows = list(reader)

    if len(raw_rows) != len(template):
        raise ValueError(
            f"{raw_path} row count differs from template: "
            f"expected {len(template)}, got {len(raw_rows)}"
        )

    seen_ids: set[str] = set()
    canonical_rows: list[dict[str, str]] = []
    counts = Counter[str]()
    format_counts = Counter[str]()
    decision_tokens = Counter[str]()
    invalid_ids: set[str] = set()
    for raw in raw_rows:
        identifier = raw["id"]
        expected_identity = template_by_id.get(identifier)
        if expected_identity is None:
            raise ValueError(f"{raw_path} contains unknown ID {identifier!r}")
        if identifier in seen_ids:
            raise ValueError(f"{raw_path} contains duplicate ID {identifier!r}")
        seen_ids.add(identifier)
        if source_identity(raw) != expected_identity:
            raise ValueError(f"{raw_path} mutated source fields for {identifier}")

        decision, greeting, input_format, raw_token, invalid = canonical_decision(
            raw,
            expected_identity.display_name,
        )
        format_counts[input_format] += 1
        decision_tokens[raw_token] += 1
        if decision == "GREETING":
            counts["exact_greeting"] += 1
        elif decision == "NULL":
            counts["null"] += 1
        elif invalid:
            counts["invalid_or_empty_mapped_to_skip"] += 1
            invalid_ids.add(identifier)
        else:
            counts["annotator_skip"] += 1
        canonical_rows.append(
            {
                "id": identifier,
                "display_name": expected_identity.display_name,
                "country_hint": expected_identity.country_hint,
                "locale_hint": expected_identity.locale_hint,
                "decision": decision,
                "expected_greeting": greeting,
            }
        )

    if seen_ids != set(template_by_id):
        raise ValueError(f"{raw_path} does not cover every template ID")
    metadata: dict[str, object] = {
        "rows": len(raw_rows),
        "input_formats": dict(sorted(format_counts.items())),
        "raw_decision_tokens": dict(sorted(decision_tokens.items())),
        "exact_greeting": counts["exact_greeting"],
        "null": counts["null"],
        "annotator_skip": counts["annotator_skip"],
        "invalid_or_empty_mapped_to_skip": counts[
            "invalid_or_empty_mapped_to_skip"
        ],
    }
    canonical_rows.sort(key=lambda row: row["id"])
    return canonical_rows, metadata, invalid_ids


def canonical_decision(
    row: dict[str, str],
    display_name: str,
) -> tuple[str, str, str, str, bool]:
    raw_decision = row.get("decision", "")
    normalized_decision = raw_decision.strip().upper()
    raw_token = normalized_decision or "<empty>"
    if normalized_decision:
        greeting = row.get("expected_greeting", "")
        if normalized_decision in {"GREETING", "GREET"}:
            if greeting and greeting in display_name:
                return "GREETING", greeting, "canonical", raw_token, False
            return "SKIP", "", "canonical", raw_token, True
        if normalized_decision == "NULL" and not greeting:
            return "NULL", "", "canonical", raw_token, False
        if normalized_decision == "SKIP":
            return "SKIP", "", "canonical", raw_token, False
        return "SKIP", "", "canonical", raw_token, True

    if "first_name" not in row:
        return "SKIP", "", "canonical", raw_token, True
    first_name = row["first_name"]
    normalized_first_name = first_name.strip().upper()
    if normalized_first_name == "NULL":
        return "NULL", "", "legacy_first_name", raw_token, False
    if normalized_first_name == "SKIP":
        return "SKIP", "", "legacy_first_name", raw_token, False
    if first_name and first_name in display_name:
        return "GREETING", first_name, "legacy_first_name", raw_token, False
    return "SKIP", "", "legacy_first_name", raw_token, True


def serialize_canonical(rows: list[dict[str, str]]) -> bytes:
    with tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CANONICAL_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        file.seek(0)
        return file.read().encode("utf-8")


def serialize_provenance(provenance: dict[str, object]) -> bytes:
    return (
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def publish_outputs(outputs: list[tuple[Path, bytes]]) -> None:
    temporary_paths: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, value in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(raw_path)
            with os.fdopen(descriptor, "wb") as file:
                file.write(value)
                file.flush()
                os.fsync(file.fileno())
            temporary_paths.append((temporary, destination))
        for temporary, destination in temporary_paths:
            os.link(temporary, destination)
            published.append(destination)
    except BaseException:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)


def main() -> None:
    arguments = parse_args()
    annotations = [
        AnnotationInput(label, Path(raw), Path(canonical))
        for label, raw, canonical in arguments.annotation
    ]
    normalize_annotations(
        arguments.source_csv,
        arguments.template_csv,
        annotations,
        arguments.provenance_json,
    )


if __name__ == "__main__":
    main()
