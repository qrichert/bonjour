import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from morphology_nn.cli import build_parser, verify_selection
from morphology_nn.data import sha256_file


class CliTests(unittest.TestCase):
    def test_select_has_no_test_or_proxy_input(self):
        parser = build_parser()
        arguments = [
            "select",
            "--train",
            "train.csv",
            "--validation",
            "validation.csv",
            "--manifest",
            "manifest.json",
            "--output",
            "selection",
            "--test",
            "morph_test.csv",
        ]
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(arguments)

    def test_selection_receipt_binds_every_frozen_file(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            files = {
                "model_sha256": directory / "selected_model.f32.bin",
                "model_metadata_sha256": directory / "selected_model.json",
                "thresholds_sha256": directory / "frozen_thresholds.csv",
            }
            for index, path in enumerate(files.values()):
                path.write_bytes(f"fixture-{index}".encode())
            receipt = {key: sha256_file(path) for key, path in files.items()}
            (directory / "selection_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            self.assertEqual(verify_selection(directory), receipt)

            files["thresholds_sha256"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                verify_selection(directory)


if __name__ == "__main__":
    unittest.main()
