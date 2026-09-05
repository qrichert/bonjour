import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "normalize_proxy_annotations.py"
SPEC = importlib.util.spec_from_file_location(
    "normalize_proxy_annotations", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


SOURCE_ROWS = [
    ("case-00000000", "Anne Marie Dupont", "FR", "fr-FR"),
    ("case-00000001", "İbrahim Yılmaz", "TR", "tr-TR"),
    ("case-00000002", "Baris Kebab", "", ""),
    ("case-00000003", "Élodie Martin", "FR", ""),
]


def write_source(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(("display_name", "country_hint", "locale_hint"))
        for _, display_name, country_hint, locale_hint in SOURCE_ROWS:
            writer.writerow((display_name, country_hint, locale_hint))


def write_template(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(SCRIPT.CANONICAL_HEADER)
        for identifier, display_name, country_hint, locale_hint in SOURCE_ROWS:
            writer.writerow(
                (identifier, display_name, country_hint, locale_hint, "", "")
            )


def write_annotation(
    path: Path,
    extra_header: tuple[str, ...],
    values: list[tuple[str, ...]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow((*SCRIPT.IDENTITY_HEADER, *extra_header))
        for source, annotation in zip(SOURCE_ROWS, values, strict=True):
            writer.writerow((*source, *annotation))


class NormalizeProxyAnnotationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source.csv"
        self.template = self.directory / "template.csv"
        write_source(self.source)
        write_template(self.template)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def normalize(self, suffix: str):
        raw_a = self.directory / "raw-a.csv"
        raw_b = self.directory / "raw-b.csv"
        write_annotation(
            raw_a,
            ("decision", "expected_greeting", "confidence", "note"),
            [
                ("GREET", "Anne Marie", "high", "ignored"),
                ("GREETING", "İbrahim", "high", "ignored"),
                ("NULL", "", "high", "ignored"),
                ("GREET", "Elodie", "high", "not an exact accented span"),
            ],
        )
        write_annotation(
            raw_b,
            ("decision", "expected_greeting", "first_name"),
            [
                ("", "", "Anne Marie"),
                ("", "", "İbrahim"),
                ("", "", "NULL"),
                ("", "", "SKIP"),
            ],
        )
        output_directory = self.directory / suffix
        output_directory.mkdir()
        canonical_a = output_directory / "canonical-a.csv"
        canonical_b = output_directory / "canonical-b.csv"
        provenance = output_directory / "provenance.json"
        raw_hashes = (sha256(raw_a), sha256(raw_b))
        result = SCRIPT.normalize_annotations(
            self.source,
            self.template,
            [
                SCRIPT.AnnotationInput("a", raw_a, canonical_a),
                SCRIPT.AnnotationInput("b", raw_b, canonical_b),
            ],
            provenance,
        )
        self.assertEqual((sha256(raw_a), sha256(raw_b)), raw_hashes)
        return canonical_a, canonical_b, provenance, result

    def test_normalizes_canonical_and_legacy_shapes_without_repairing(self) -> None:
        canonical_a, canonical_b, provenance_path, result = self.normalize("one")

        rows_a = read_rows(canonical_a)
        rows_b = read_rows(canonical_b)
        self.assertEqual(rows_a[0]["expected_greeting"], "Anne Marie")
        self.assertEqual(rows_a[1]["expected_greeting"], "İbrahim")
        self.assertEqual(rows_a[2]["decision"], "NULL")
        self.assertEqual(rows_a[3]["decision"], "SKIP")
        self.assertEqual(rows_a[3]["expected_greeting"], "")
        self.assertEqual(rows_b[0]["decision"], "GREETING")
        self.assertEqual(rows_b[3]["decision"], "SKIP")
        self.assertEqual(result["unusable_or_non_exact_cases"], 1)
        self.assertEqual(result["annotations"][0]["exact_greeting"], 2)
        self.assertEqual(result["annotations"][0]["invalid_or_empty_mapped_to_skip"], 1)
        self.assertNotIn(
            "not an exact accented span",
            provenance_path.read_text(encoding="utf-8"),
        )

    def test_output_bytes_are_deterministic(self) -> None:
        first_a, first_b, first_provenance, _ = self.normalize("one")
        first = (
            first_a.read_bytes(),
            first_b.read_bytes(),
            first_provenance.read_bytes(),
        )
        second_a, second_b, second_provenance, _ = self.normalize("two")
        second = (
            second_a.read_bytes(),
            second_b.read_bytes(),
            second_provenance.read_bytes(),
        )
        self.assertEqual(first, second)
        parsed = json.loads(first_provenance.read_text(encoding="utf-8"))
        self.assertEqual(parsed["format_version"], 1)

    def test_rejects_mutated_source_duplicate_ids_and_missing_rows(self) -> None:
        valid = self.directory / "valid.csv"
        write_annotation(
            valid,
            ("decision", "expected_greeting"),
            [("NULL", "")] * len(SOURCE_ROWS),
        )

        mutated = self.directory / "mutated.csv"
        mutated.write_bytes(valid.read_bytes())
        text = mutated.read_text(encoding="utf-8").replace(
            "Anne Marie Dupont", "Mutated Name", 1
        )
        mutated.write_text(text, encoding="utf-8", newline="")
        with self.assertRaisesRegex(ValueError, "mutated source fields"):
            self.run_single(mutated, "mutated")

        duplicate = self.directory / "duplicate.csv"
        with valid.open(encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        rows[2][0] = rows[1][0]
        with duplicate.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            self.run_single(duplicate, "duplicate")

        missing = self.directory / "missing.csv"
        with missing.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerows(rows[:-1])
        with self.assertRaisesRegex(ValueError, "row count differs"):
            self.run_single(missing, "missing")

    def test_rejects_source_template_mismatch_and_existing_outputs(self) -> None:
        raw = self.directory / "raw.csv"
        write_annotation(
            raw,
            ("decision", "expected_greeting"),
            [("NULL", "")] * len(SOURCE_ROWS),
        )
        self.source.write_text(
            self.source.read_text(encoding="utf-8").replace("Baris", "Bariş", 1),
            encoding="utf-8",
            newline="",
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.run_single(raw, "source")

        write_source(self.source)
        output = self.directory / "canonical-existing.csv"
        output.write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            SCRIPT.normalize_annotations(
                self.source,
                self.template,
                [SCRIPT.AnnotationInput("a", raw, output)],
                self.directory / "existing-provenance.json",
            )
        self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def run_single(self, raw: Path, suffix: str) -> None:
        SCRIPT.normalize_annotations(
            self.source,
            self.template,
            [
                SCRIPT.AnnotationInput(
                    "a", raw, self.directory / f"canonical-{suffix}.csv"
                )
            ],
            self.directory / f"provenance-{suffix}.json",
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


if __name__ == "__main__":
    unittest.main()
