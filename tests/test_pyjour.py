import contextlib
import dataclasses
import io
import json
import os
import subprocess
import unittest

import pyjour
from pyjour.main import main


class InferenceTests(unittest.TestCase):
    def test_public_inference_contract(self) -> None:
        inference = pyjour.infer("Quentin Richert")

        self.assertEqual(inference.best_candidate, "Quentin")
        self.assertEqual(inference.greeting_name, "Quentin")
        self.assertEqual(inference.emission_source, "c3_1")
        self.assertEqual(inference.gender_hint, "male")
        self.assertGreater(inference.decision_score, 0.0)
        self.assertGreater(inference.gender_confidence, 0.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inference.greeting_name = None  # type: ignore[misc]

    def test_abstention_keeps_the_best_candidate_separate(self) -> None:
        inference = pyjour.infer("Martin Emmanuel")

        self.assertEqual(inference.best_candidate, "Martin Emmanuel")
        self.assertIsNone(inference.greeting_name)
        self.assertEqual(inference.emission_source, "abstain")
        self.assertIsNone(inference.gender_hint)
        self.assertEqual(inference.gender_confidence, 0.0)

    def test_unicode_round_trip_preserves_the_exact_source_span(self) -> None:
        display_name = "E\u0301lodie Durand"
        inference = pyjour.infer(display_name, country_hint="FR")

        self.assertEqual(inference.best_candidate, "E\u0301lodie")
        self.assertEqual(inference.greeting_name, "E\u0301lodie")
        self.assertIn(inference.greeting_name, display_name)

    def test_country_locale_and_gender_hints_match_rust_semantics(self) -> None:
        locale = pyjour.infer(
            "Quentin Richert", country_hint="invalid", locale_hint="fr_FR"
        )
        country = pyjour.infer("Quentin Richert", country_hint=" fr ")
        self.assertEqual(locale, country)

        male = pyjour.infer("Simone", gender_hint="M")
        female = pyjour.infer("Simone", gender_hint="female")
        self.assertEqual(male.gender_hint, "male")
        self.assertEqual(female.gender_hint, "female")

        with self.assertRaisesRegex(ValueError, "gender_hint must be"):
            pyjour.infer("Simone", gender_hint="unknown")

    def test_detailed_shape_is_stable(self) -> None:
        detailed = pyjour.infer_detailed("Quentin Richert")

        self.assertEqual(
            list(detailed),
            [
                "input",
                "best_candidate",
                "greeting_name",
                "decision_score",
                "decision",
                "candidates",
                "gender_hint",
                "gender_confidence",
            ],
        )
        self.assertEqual(detailed["input"], "Quentin Richert")
        self.assertEqual(detailed["decision"]["emission_source"], "c3_1")
        self.assertEqual(detailed["candidates"][0]["candidate"], "Quentin")

    @unittest.skipUnless(
        os.environ.get("BONJOUR_RUST_CLI"), "Rust CLI path not supplied"
    )
    def test_detailed_output_matches_the_rust_cli(self) -> None:
        display_name = "Élodie Durand"
        rust = subprocess.run(
            [os.environ["BONJOUR_RUST_CLI"], "--json", "--country=FR", display_name],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            pyjour.infer_detailed(display_name, country_hint="FR"),
            json.loads(rust.stdout),
        )


class CommandLineTests(unittest.TestCase):
    def test_plain_greeting_and_fallback(self) -> None:
        self.assertEqual(run_main(["Quentin Richert"]), "Bonjour Quentin !\n")
        self.assertEqual(
            run_main(["Quentin Richert SAS"]),
            "Bonjour Quentin Richert SAS !\n",
        )

    def test_json_output_is_unicode_and_machine_readable(self) -> None:
        output = run_main(["--json", "Élodie Durand"])
        detailed = json.loads(output)

        self.assertIn("Élodie", output)
        self.assertEqual(detailed, pyjour.infer_detailed("Élodie Durand"))

    def test_threshold_option_does_not_exist(self) -> None:
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(SystemExit, "2"),
        ):
            main(["--threshold=0.8", "Quentin Richert"])

    def test_version_matches_the_distribution(self) -> None:
        self.assertEqual(pyjour.__version__, "0.1.0")
        with (
            contextlib.redirect_stdout(io.StringIO()) as stdout,
            self.assertRaisesRegex(SystemExit, "0"),
        ):
            main(["--version"])
        self.assertEqual(stdout.getvalue(), "pyjour 0.1.0\n")


def run_main(arguments: list[str]) -> str:
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        main(arguments)
    return stdout.getvalue()


if __name__ == "__main__":
    unittest.main()
