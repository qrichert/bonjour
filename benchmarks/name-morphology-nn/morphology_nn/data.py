from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
TRUNCATED_MIDDLE_ID = 3
BYTE_OFFSET = 4
BYTE_VOCABULARY_SIZE = 260
MAX_IDS = 96
MAX_UNTRUNCATED_BYTES = MAX_IDS - 2


@dataclass(frozen=True)
class Example:
    normalized: str
    family: str
    split: str
    label: int
    populations: tuple[str, ...]
    given_count: int
    surname_count: int
    role_llr: float
    role_signal: float
    primary_country: str
    production_candidate_quality: float | None
    production_role_llr: float | None
    production_role_signal: float | None
    production_reliability: float | None
    script: str
    utf8_bytes: int
    truncated: bool


@dataclass(frozen=True)
class TensorData:
    examples: tuple[Example, ...]
    ids: torch.Tensor
    mask: torch.Tensor
    countries: torch.Tensor
    labels: torch.Tensor


def load_examples(path: Path) -> tuple[Example, ...]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "normalized",
            "family",
            "split",
            "label",
            "populations",
            "given_count",
            "surname_count",
            "role_llr",
            "role_signal",
            "primary_country",
            "production_candidate_quality",
            "production_role_llr",
            "production_role_signal",
            "production_reliability",
            "script",
            "utf8_bytes",
            "truncated",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"unexpected dataset header in {path}")
        examples = tuple(_example(row) for row in reader)
    if not examples:
        raise ValueError(f"empty dataset: {path}")
    if len({example.normalized for example in examples}) != len(examples):
        raise ValueError(f"duplicate normalized strings in {path}")
    return examples


def country_vocabulary(examples: Iterable[Example]) -> tuple[str, ...]:
    return ("UNKNOWN",) + tuple(
        sorted(
            {example.primary_country for example in examples if example.primary_country}
        )
    )


def tensorize(examples: tuple[Example, ...], countries: tuple[str, ...]) -> TensorData:
    country_ids = {country: index for index, country in enumerate(countries)}
    encoded = [encode_utf8(example.normalized) for example in examples]
    ids = torch.tensor([value[0] for value in encoded], dtype=torch.long)
    mask = torch.tensor([value[1] for value in encoded], dtype=torch.bool)
    country = torch.tensor(
        [country_ids.get(example.primary_country, 0) for example in examples],
        dtype=torch.long,
    )
    labels = torch.tensor([example.label for example in examples], dtype=torch.float32)
    return TensorData(examples, ids, mask, country, labels)


def encode_utf8(value: str) -> tuple[list[int], list[bool], bool]:
    payload = value.encode("utf-8")
    truncated = len(payload) > MAX_UNTRUNCATED_BYTES
    if truncated:
        retained = MAX_IDS - 3
        prefix = (retained + 1) // 2
        suffix = retained - prefix
        tokens = (
            [BOS_ID]
            + [byte + BYTE_OFFSET for byte in payload[:prefix]]
            + [TRUNCATED_MIDDLE_ID]
            + [byte + BYTE_OFFSET for byte in payload[-suffix:]]
            + [EOS_ID]
        )
    else:
        tokens = [BOS_ID] + [byte + BYTE_OFFSET for byte in payload] + [EOS_ID]
    mask = [True] * len(tokens)
    padding = MAX_IDS - len(tokens)
    tokens.extend([PAD_ID] * padding)
    mask.extend([False] * padding)
    return tokens, mask, truncated


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_file(path: Path, manifest: dict) -> None:
    expected = manifest["output_sha256"].get(path.name)
    if expected is None:
        raise ValueError(f"{path.name} is not recorded in dataset manifest")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"dataset checksum mismatch for {path}: {actual} != {expected}"
        )


def _example(row: dict[str, str]) -> Example:
    return Example(
        normalized=row["normalized"],
        family=row["family"],
        split=row["split"],
        label=int(row["label"]),
        populations=tuple(filter(None, row["populations"].split(";"))),
        given_count=int(row["given_count"]),
        surname_count=int(row["surname_count"]),
        role_llr=float(row["role_llr"]),
        role_signal=float(row["role_signal"]),
        primary_country=row["primary_country"],
        production_candidate_quality=_optional_float(
            row["production_candidate_quality"]
        ),
        production_role_llr=_optional_float(row["production_role_llr"]),
        production_role_signal=_optional_float(row["production_role_signal"]),
        production_reliability=_optional_float(row["production_reliability"]),
        script=row["script"],
        utf8_bytes=int(row["utf8_bytes"]),
        truncated=row["truncated"] == "true",
    )


def _optional_float(value: str) -> float | None:
    return float(value) if value else None
