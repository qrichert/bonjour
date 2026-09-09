import io
import unittest
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

from morphology_nn.cli import build_parser
from morphology_nn.conditional import (
    ConditionalRow,
    Policy,
    RawRow,
    build_conditional_buckets,
    build_masks,
    build_same_error_rows,
    counts_for_mask,
    materialize_rows,
    read_conditional_rows,
    recommend,
    retain_representative,
    select_at_budget,
    validate_raw_row,
    verify_frozen_inputs,
)


class ConditionalTests(unittest.TestCase):
    def test_cli_accepts_receipt_but_not_morphology_test_rows(self):
        parser = build_parser()
        arguments = [
            "evaluate-conditional",
            "--selection",
            "selection",
            "--test-receipt",
            "test_receipt.json",
            "--output",
            "conditional",
        ]
        parsed = parser.parse_args(arguments)
        self.assertEqual(str(parsed.test_receipt), "test_receipt.json")
        self.assertFalse(hasattr(parsed, "test"))

    def test_stream_rejects_an_unexpected_schema(self):
        with self.assertRaisesRegex(ValueError, "unexpected conditional proxy header"):
            read_conditional_rows(io.StringIO("normalized_candidate\nalice\n"))

    def test_neural_threshold_is_conditional_on_the_same_baseline_gate(self):
        rows = (
            row(selected_matches=True, neural_score=0.95),
            row(expected_greeting=True, neural_score=0.10),
            row(
                selected_matches=True,
                candidate_count=2,
                winner_margin=0.20,
                neural_score=0.90,
            ),
            row(selected_matches=True, c4_emits=True, eligible=False),
        )
        masks = build_masks(rows)
        baseline_signature = (
            masks.quality[0.7]
            & masks.reliability[0.7]
            & masks.role[0.5]
            & (masks.sole | (masks.multiple & masks.margin[0.3]))
        )
        augmented_signature = baseline_signature & masks.neural[0.9]
        baseline = Policy(
            "baseline", 0.7, 0.7, 0.5, 0.3, None, baseline_signature, baseline_signature
        )
        augmented = Policy(
            "augmented",
            0.7,
            0.7,
            0.5,
            0.3,
            0.9,
            augmented_signature,
            baseline_signature,
        )

        self.assertEqual(
            counts_for_mask(baseline.signature, masks, masks.all_rows).correct, 1
        )
        self.assertEqual(
            counts_for_mask(baseline.signature, masks, masks.all_rows).errors, 1
        )
        self.assertEqual(
            counts_for_mask(augmented.signature, masks, masks.all_rows).correct, 1
        )
        self.assertEqual(
            counts_for_mask(augmented.signature, masks, masks.all_rows).errors, 0
        )
        self.assertFalse(augmented.signature & masks.c4)
        self.assertEqual(masks.neural[0.0], masks.eligible)

    def test_thresholds_are_inclusive_monotonic_and_margin_is_topology_specific(self):
        rows = (
            row(
                selected_matches=True,
                candidate_quality=0.7,
                reliability=0.7,
                role_signal=0.5,
                neural_score=0.9,
            ),
            row(
                selected_matches=True,
                candidate_count=2,
                winner_margin=0.3,
                candidate_quality=0.8,
                reliability=0.8,
                role_signal=0.6,
                neural_score=0.95,
            ),
        )
        masks = build_masks(rows)
        self.assertEqual(masks.quality[0.7], 0b11)
        self.assertEqual(masks.reliability[0.7], 0b11)
        self.assertEqual(masks.role[0.5], 0b11)
        self.assertEqual(masks.neural[0.9], 0b11)
        self.assertEqual(masks.multiple & masks.margin[0.3], 0b10)
        low_margin = masks.sole | (masks.multiple & masks.margin[0.0])
        high_margin = masks.sole | (masks.multiple & masks.margin[1.0])
        self.assertEqual(low_margin & masks.sole, high_margin & masks.sole)
        self.assertNotEqual(low_margin, high_margin)
        for thresholds in (
            masks.quality,
            masks.reliability,
            masks.role,
            masks.margin,
            masks.neural,
        ):
            ordered = list(thresholds.values())
            for lower, higher in pairwise(ordered):
                self.assertEqual(higher & ~lower, 0)

    def test_materialization_enforces_native_veto_and_c4_exclusions(self):
        raw_rows = (
            raw_row(),
            raw_row(native_candidate=False, segmented_candidate=True),
            raw_row(vetoes_pass=False, hard_organization_marker=True),
            raw_row(c4_emits=True, c4_source="sole_native", c5_emits=True),
        )
        rows = materialize_rows(raw_rows, [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(
            [value.eligible for value in rows], [True, False, False, False]
        )
        self.assertNotIn("normalized_candidate", ConditionalRow.__dataclass_fields__)

        invalid = replace(raw_rows[0], vetoes_pass=False)
        with self.assertRaisesRegex(ValueError, "veto fields disagree"):
            validate_raw_row(invalid)

    def test_bitset_counts_match_naive_outcomes(self):
        rows = (
            row(selected_matches=True),
            row(expected_greeting=True),
            row(),
        )
        masks = build_masks(rows)
        signature = 0b111
        counts = counts_for_mask(signature, masks, masks.all_rows)
        self.assertEqual(counts.correct, sum(value.selected_matches for value in rows))
        self.assertEqual(
            counts.wrong_winner,
            sum(
                value.expected_greeting and not value.selected_matches for value in rows
            ),
        )
        self.assertEqual(
            counts.null_winner,
            sum(not value.expected_greeting for value in rows),
        )

    def test_duplicate_signatures_keep_the_stricter_representative(self):
        policies = {}
        lower = Policy("baseline", 0.5, 0.5, 0.5, 0.5, None, 1, 1)
        higher = Policy("baseline", 0.7, 0.6, 0.5, 0.5, None, 1, 1)
        retain_representative(policies, lower)
        retain_representative(policies, higher)
        retain_representative(policies, lower)
        self.assertEqual(policies, {1: higher})

    def test_same_error_dominance_requires_binding_nonrepresentable_signature(self):
        rows = (
            row(selected_matches=True),
            row(selected_matches=True),
            row(expected_greeting=True),
        )
        masks = build_masks(rows)
        baseline = Policy("baseline", 0.7, 0.7, 0.7, 0.7, None, 0b001, 0b001)
        augmented = Policy("augmented", 0.5, 0.5, 0.5, 0.5, 0.9, 0b011, 0b111)
        comparison = build_same_error_rows(
            [(baseline, counts_for_mask(baseline.signature, masks, masks.all_rows))],
            [(augmented, counts_for_mask(augmented.signature, masks, masks.all_rows))],
            frozenset((0b001, 0b111)),
            masks,
        )
        self.assertEqual(comparison[0]["correct_gain"], 1)
        self.assertTrue(comparison[0]["strict_augmented_domination"])

    def test_training_scope_cannot_count_a_heldout_error(self):
        rows = (
            row(expected_greeting=True, population="REAL_PROXY_V1_DEV"),
            row(selected_matches=True, population="REAL_PROXY_V3"),
        )
        masks = build_masks(rows)
        selected = Policy("baseline", 0.0, 0.0, 0.0, 0.0, None, 0b11, 0b11)
        empty = Policy("baseline", 1.0, 1.0, 1.0, 1.0, None, 0, 0)
        training = masks.all_rows & ~masks.populations["REAL_PROXY_V1_DEV"]
        training_policy, _ = select_at_budget((selected, empty), masks, training, 0)
        full_policy, _ = select_at_budget((selected, empty), masks, masks.all_rows, 0)
        self.assertEqual(training_policy, selected)
        self.assertEqual(full_policy, empty)

    def test_conditional_buckets_use_frozen_bands_and_suppress_sparse_rows(self):
        rows = tuple(
            row(
                selected_matches=index < 4,
                expected_greeting=index == 4,
                neural_score=0.9,
            )
            for index in range(5)
        ) + (row(selected_matches=True, neural_score=0.98),)
        buckets, suppressed_rows, suppressed_strata = build_conditional_buckets(rows)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0]["quality_bin"], "[0.75,0.8)")
        self.assertEqual(buckets[0]["neural_bin"], "[0.9,0.975)")
        self.assertEqual(buckets[0]["correct"], 4)
        self.assertEqual(buckets[0]["wrong_winner"], 1)
        self.assertEqual((suppressed_rows, suppressed_strata), (1, 1))

    def test_frozen_selection_hash_is_rejected_before_scoring(self):
        with (
            patch(
                "morphology_nn.conditional.verify_selection",
                return_value={"model_sha256": "changed"},
            ),
            self.assertRaisesRegex(ValueError, "frozen selection changed"),
        ):
            verify_frozen_inputs(Path("selection"), Path("test_receipt.json"))

    def test_recommendation_requires_pooled_and_stable_oof_evidence(self):
        pooled = [
            {
                "error_budget": 1,
                "strict_augmented_domination": True,
                "correct_gain": 3,
            }
        ]
        stable = [
            {
                "error_budget": 1,
                "correct_gain": 2,
                "error_delta": 0,
                "all_selected_nn_binding": True,
                "all_selected_nn_nonrepresentable": True,
                "no_heldout_generation_adds_errors": True,
                "no_heldout_generation_loses_correct": True,
                "v1_correct_gain": 2,
                "v3_correct_gain": 0,
                "v4_correct_gain": 0,
            }
        ]
        recommendation, _ = recommend(pooled, stable)
        self.assertEqual(
            recommendation, "proceed to a classifier-integration experiment"
        )

        stable[0]["no_heldout_generation_loses_correct"] = False
        stable[0]["v3_correct_gain"] = -1
        recommendation, _ = recommend(pooled, stable)
        self.assertEqual(
            recommendation,
            "conditional signal, not integration-ready",
        )


def row(
    *,
    expected_greeting: bool = False,
    selected_matches: bool = False,
    c4_emits: bool = False,
    candidate_count: int = 1,
    eligible: bool = True,
    winner_margin: float = 1.0,
    neural_score: float = 0.5,
    candidate_quality: float = 0.75,
    reliability: float = 0.8,
    role_signal: float = 0.6,
    population: str = "REAL_PROXY_V1_DEV",
) -> ConditionalRow:
    return ConditionalRow(
        population=population,
        expected_greeting=expected_greeting or selected_matches,
        selected_matches=selected_matches,
        winner_present=True,
        c4_emits=c4_emits,
        c5_emits=c4_emits,
        candidate_count=candidate_count,
        eligible=eligible,
        role_signal=role_signal,
        reliability=reliability,
        winner_margin=winner_margin,
        candidate_quality=candidate_quality,
        neural_score=neural_score,
    )


def raw_row(
    *,
    native_candidate: bool = True,
    segmented_candidate: bool = False,
    vetoes_pass: bool = True,
    hard_organization_marker: bool = False,
    c4_emits: bool = False,
    c4_source: str = "abstain",
    c5_emits: bool = False,
) -> RawRow:
    return RawRow(
        population="REAL_PROXY_V1_DEV",
        normalized_candidate="alice",
        country_hint="FR",
        expected_greeting=True,
        selected_matches=True,
        winner_present=True,
        c31_emits=False,
        c4_emits=c4_emits,
        c4_source=c4_source,
        c5_emits=c5_emits,
        candidate_count=1,
        native_candidate=native_candidate,
        segmented_candidate=segmented_candidate,
        vetoes_pass=vetoes_pass,
        hard_organization_marker=hard_organization_marker,
        generic_organization_marker=False,
        ampersand=False,
        candidate_too_short=False,
        role_signal=0.6,
        reliability=0.8,
        winner_margin=1.0,
        candidate_quality=0.75,
    )


if __name__ == "__main__":
    unittest.main()
