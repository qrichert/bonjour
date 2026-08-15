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


def write_holdout_source(path: Path, names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(SCRIPT.OUTPUT_HEADER)
        writer.writerows((name, "", "") for name in names)


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

    def test_exact_exclusions_are_deterministic_and_preserve_remaining_duplicates(
        self,
    ) -> None:
        write_users(self.source, ["V1", "V1", "Repeat", "Repeat", "Other", "Third"])
        exclusion = self.directory / "v1-source.csv"
        write_holdout_source(exclusion, ["V1", "V1"])

        first_output = self.directory / "v2-one.csv"
        first_provenance = self.directory / "v2-one.json"
        second_output = self.directory / "v2-two.csv"
        second_provenance = self.directory / "v2-two.json"
        source_before = hashlib.sha256(self.source.read_bytes()).hexdigest()

        first = SCRIPT.prepare(
            self.source,
            first_output,
            first_provenance,
            sample_size=4,
            rng_seed=0x5632,
            exclude_source_paths=[exclusion],
        )
        SCRIPT.prepare(
            self.source,
            second_output,
            second_provenance,
            sample_size=4,
            rng_seed=0x5632,
            exclude_source_paths=[exclusion],
        )

        self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
        with first_output.open(encoding="utf-8", newline="") as source:
            names = [row["display_name"] for row in csv.DictReader(source)]
        self.assertNotIn("V1", names)
        self.assertEqual(names.count("Repeat"), 2)
        self.assertEqual(first["nonblank_rows_before_exact_exclusions"], 6)
        self.assertEqual(first["excluded_exact_display_name_rows"], 2)
        self.assertEqual(first["excluded_unique_display_names"], 1)
        self.assertEqual(first["eligible_nonblank_rows"], 4)
        self.assertEqual(first["exclusion_sources"][0]["rows"], 2)
        self.assertEqual(first["exclusion_sources"][0]["unique_display_names"], 1)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), source_before)

    def test_rejects_bad_exclusion_schema_and_insufficient_remaining_population(
        self,
    ) -> None:
        write_users(self.source, ["One", "Two", "Three"])
        exclusion = self.directory / "exclusion.csv"
        exclusion.write_text("display_name\nOne\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported header"):
            SCRIPT.prepare(
                self.source,
                self.directory / "bad-output.csv",
                self.directory / "bad-provenance.json",
                sample_size=1,
                exclude_source_paths=[exclusion],
            )

        write_holdout_source(exclusion, ["One", "Two"])
        with self.assertRaisesRegex(ValueError, "only 1 nonblank"):
            SCRIPT.prepare(
                self.source,
                self.directory / "small-output.csv",
                self.directory / "small-provenance.json",
                sample_size=2,
                exclude_source_paths=[exclusion],
            )


if __name__ == "__main__":
    unittest.main()
