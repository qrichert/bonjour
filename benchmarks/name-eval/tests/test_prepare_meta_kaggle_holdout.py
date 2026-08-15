import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_meta_kaggle_holdout.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_meta_kaggle_holdout", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def write_users(path: Path, names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(SCRIPT.EXPECTED_HEADER)
        for index, name in enumerate(names, start=1):
            writer.writerow(
                (
                    index,
                    f"user{index}",
                    name,
                    "01/01/2020",
                    1,
                    "France",
                    "False",
                    "False",
                )
            )


class PrepareMetaKaggleHoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "Users.csv"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(self, suffix: str, sample_size: int = 4, seed: int = 7):
        output = self.directory / f"source-{suffix}.csv"
        provenance = self.directory / f"provenance-{suffix}.json"
        result = SCRIPT.prepare(self.source, output, provenance, sample_size, seed)
        return output, provenance, result

    def test_output_is_deterministic_minimal_and_unicode_preserving(self) -> None:
        names = ["Élodie", "Anne Marie", "O'Connor", "同名", "Repeat", "Repeat"]
        write_users(self.source, names)
        source_before = hashlib.sha256(self.source.read_bytes()).hexdigest()

        first, first_provenance, result = self.prepare("one")
        second, _, _ = self.prepare("two")

        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(
            hashlib.sha256(self.source.read_bytes()).hexdigest(), source_before
        )
        with first.open(encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        self.assertEqual(rows[0], list(SCRIPT.OUTPUT_HEADER))
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row[1:] == ["", ""] for row in rows[1:]))
        self.assertTrue(all(row[0] in names for row in rows[1:]))
        self.assertEqual(result["source_sha256_before"], source_before)
        self.assertEqual(result["source_sha256_after"], source_before)

        provenance_text = first_provenance.read_text(encoding="utf-8")
        for name in names:
            self.assertNotIn(name, provenance_text)

    def test_only_blank_names_are_excluded_and_duplicates_are_rows(self) -> None:
        write_users(self.source, ["", "   ", "Repeat", "Repeat", "Other"])
        output, provenance_path, _ = self.prepare("blanks", sample_size=3)

        with output.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["display_name"] for row in rows].count("Repeat"), 2)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["source_rows"], 5)
        self.assertEqual(provenance["eligible_nonblank_rows"], 3)
        self.assertEqual(provenance["excluded_blank_or_whitespace_rows"], 2)

    def test_refuses_to_overwrite_either_output(self) -> None:
        write_users(self.source, ["One", "Two"])
        output, provenance, _ = self.prepare("initial", sample_size=1)

        with self.assertRaises(FileExistsError):
            SCRIPT.prepare(
                self.source,
                output,
                self.directory / "new-provenance.json",
                1,
                7,
            )
        with self.assertRaises(FileExistsError):
            SCRIPT.prepare(
                self.source,
                self.directory / "new-output.csv",
                provenance,
                1,
                7,
            )

    def test_rejects_changed_header_and_undersized_population(self) -> None:
        self.source.write_text(
            "DisplayName,Country\nExample,France\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "header changed"):
            self.prepare("header", sample_size=1)

        write_users(self.source, ["", "Only One"])
        with self.assertRaisesRegex(ValueError, "only 1 nonblank"):
            self.prepare("small", sample_size=2)


if __name__ == "__main__":
    unittest.main()
