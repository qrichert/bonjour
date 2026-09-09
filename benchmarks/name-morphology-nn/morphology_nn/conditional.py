from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch

from .cli import BATCH_SIZE, publish_directory, verify_selection
from .data import encode_utf8, load_json, sha256_file
from .metrics import sigmoid
from .model import load_export

MODEL_SHA256 = "09bb721a8cd44354308ee7bd83fc5f217b538647eac68d46ea95b0be65e36c45"
METADATA_SHA256 = "fb58089982e176d3f27e358c82d91f9f3633c5309aee04e3d7f0d8dba0c5d9c5"
THRESHOLDS_SHA256 = "85e4dfe18a846ff7071773e13a3ff6017a1a0fc40ab974ec780ce77fae920d31"
TEST_RECEIPT_SHA256 = "8a879bb5928473c20f15952357bfded167c2561ef5c3d2bc5c3cfe14e7964bef"
PROXY_DIGESTS = {
    "REAL_PROXY_V1_DEV": "de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e",
    "REAL_PROXY_V3": "d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe",
    "REAL_PROXY_V4": "d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f",
}
POPULATIONS = tuple(PROXY_DIGESTS)
EVIDENCE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
MARGIN_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
NEURAL_GRID = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995)
LOGO_ERROR_BUDGETS = (0, 1, 2, 3, 5, 10, 20, 30, 50)
HIGHLIGHT_ERROR_BUDGETS = frozenset(LOGO_ERROR_BUDGETS)
MINIMUM_BUCKET_ROWS = 5
CONDITIONAL_FIELDS = (
    "population",
    "ordinal",
    "normalized_candidate",
    "expected_greeting",
    "selected_matches",
    "winner_present",
    "c31_emits",
    "c4_emits",
    "c4_source",
    "c5_emits",
    "candidate_count",
    "native_candidate",
    "segmented_candidate",
    "vetoes_pass",
    "hard_organization_marker",
    "generic_organization_marker",
    "ampersand",
    "candidate_too_short",
    "role_signal",
    "reliability",
    "winner_margin",
    "candidate_quality",
    "country_hint",
)


@dataclass(frozen=True)
class RawRow:
    population: str
    normalized_candidate: str
    country_hint: str
    expected_greeting: bool
    selected_matches: bool
    winner_present: bool
    c31_emits: bool
    c4_emits: bool
    c4_source: str
    c5_emits: bool
    candidate_count: int | None
    native_candidate: bool
    segmented_candidate: bool | None
    vetoes_pass: bool
    hard_organization_marker: bool
    generic_organization_marker: bool
    ampersand: bool
    candidate_too_short: bool
    role_signal: float | None
    reliability: float | None
    winner_margin: float | None
    candidate_quality: float | None


@dataclass(frozen=True)
class ConditionalRow:
    population: str
    expected_greeting: bool
    selected_matches: bool
    winner_present: bool
    c4_emits: bool
    c5_emits: bool
    candidate_count: int
    eligible: bool
    role_signal: float
    reliability: float
    winner_margin: float
    candidate_quality: float
    neural_score: float


@dataclass(frozen=True)
class Masks:
    all_rows: int
    populations: dict[str, int]
    c4: int
    c5: int
    eligible: int
    correct: int
    wrong_winner: int
    null_winner: int
    sole: int
    multiple: int
    quality: dict[float, int]
    reliability: dict[float, int]
    role: dict[float, int]
    margin: dict[float, int]
    neural: dict[float, int]


@dataclass(frozen=True)
class Policy:
    family: str
    quality: float
    reliability: float
    role: float
    margin: float
    neural: float | None
    signature: int
    parent_signature: int


@dataclass(frozen=True)
class Counts:
    correct: int
    wrong_winner: int
    null_winner: int

    @property
    def errors(self) -> int:
        return self.wrong_winner + self.null_winner

    @property
    def emitted(self) -> int:
        return self.correct + self.errors


@dataclass(frozen=True)
class Analysis:
    population_summary: list[dict[str, Any]]
    conditional_buckets: list[dict[str, Any]]
    full_frontiers: list[dict[str, Any]]
    same_error: list[dict[str, Any]]
    logo_folds: list[dict[str, Any]]
    logo_oof: list[dict[str, Any]]
    recommendation: str
    recommendation_reason: str
    suppressed_bucket_rows: int
    suppressed_bucket_strata: int


def evaluate_conditional(
    *,
    selection: Path,
    test_receipt: Path,
    output: Path,
    source: TextIO,
) -> None:
    selection_receipt, model, metadata = verify_frozen_inputs(selection, test_receipt)
    raw_rows = read_conditional_rows(source)
    neural_scores = score_raw_rows(model, metadata, raw_rows)
    rows = materialize_rows(raw_rows, neural_scores)
    del raw_rows

    first = build_analysis(rows)
    second = build_analysis(rows)
    first_outputs = render_analysis(first, selection_receipt)
    second_outputs = render_analysis(second, selection_receipt)
    if first_outputs != second_outputs:
        raise ValueError("conditional analysis failed deterministic replay")

    output_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in first_outputs.items()
    }
    receipt = {
        "format_version": 1,
        "model_sha256": MODEL_SHA256,
        "model_metadata_sha256": METADATA_SHA256,
        "thresholds_sha256": THRESHOLDS_SHA256,
        "test_receipt_sha256": TEST_RECEIPT_SHA256,
        "accepted_proxy_digests": PROXY_DIGESTS,
        "rows": len(rows),
        "rows_with_winner": sum(row.winner_present for row in rows),
        "eligible_c4_abstentions": sum(row.eligible for row in rows),
        "morphology_test_rows_read": 0,
        "deterministic_replay_equal": True,
        "error_definition": "wrong winner plus expected-NULL winner",
        "grids": {
            "quality_reliability_role": EVIDENCE_GRID,
            "multiple_candidate_margin": MARGIN_GRID,
            "neural_score": NEURAL_GRID,
            "logo_error_budgets": LOGO_ERROR_BUDGETS,
        },
        "output_sha256": output_hashes,
    }
    first_outputs["conditional_receipt.json"] = render_json(receipt)

    def write(temporary: Path) -> None:
        for name, payload in first_outputs.items():
            (temporary / name).write_bytes(payload)

    publish_directory(output, write)


def verify_frozen_inputs(
    selection: Path, test_receipt: Path
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any]]:
    receipt = verify_selection(selection)
    expected_selection = {
        "model_sha256": MODEL_SHA256,
        "model_metadata_sha256": METADATA_SHA256,
        "thresholds_sha256": THRESHOLDS_SHA256,
    }
    for key, expected in expected_selection.items():
        if receipt.get(key) != expected:
            raise ValueError(f"frozen selection changed: {key}")
    if receipt.get("selected_config") != "small_country":
        raise ValueError("frozen selection is not small_country")
    if receipt.get("selected_epoch") != 20:
        raise ValueError("frozen selection epoch changed")
    if sha256_file(test_receipt) != TEST_RECEIPT_SHA256:
        raise ValueError("frozen morphology TEST receipt changed")
    test = load_json(test_receipt)
    if test.get("model_sha256") != MODEL_SHA256:
        raise ValueError("morphology TEST receipt is bound to another model")
    if test.get("thresholds_sha256") != THRESHOLDS_SHA256:
        raise ValueError("morphology TEST receipt is bound to other thresholds")

    model, metadata = load_export(selection / "selected_model.f32.bin")
    if metadata.get("parameter_count") != 13_505:
        raise ValueError("frozen model parameter count changed")
    if metadata.get("config", {}).get("name") != "small_country":
        raise ValueError("frozen model architecture changed")
    return receipt, model, metadata


def read_conditional_rows(source: TextIO) -> tuple[RawRow, ...]:
    reader = csv.DictReader(source)
    if tuple(reader.fieldnames or ()) != CONDITIONAL_FIELDS:
        raise ValueError("unexpected conditional proxy header")
    rows = []
    ordinals: dict[str, set[int]] = defaultdict(set)
    for record in reader:
        population = record["population"]
        if population not in PROXY_DIGESTS:
            raise ValueError(f"unexpected proxy population: {population}")
        ordinal = int(record["ordinal"])
        if ordinal in ordinals[population]:
            raise ValueError(f"duplicate ordinal in {population}: {ordinal}")
        ordinals[population].add(ordinal)
        row = parse_raw_row(record)
        validate_raw_row(row)
        rows.append(row)
    if set(ordinals) != set(POPULATIONS):
        raise ValueError("conditional stream must contain V1/V3/V4")
    if not rows:
        raise ValueError("empty conditional proxy stream")
    return tuple(rows)


def score_raw_rows(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    rows: tuple[RawRow, ...],
) -> list[float | None]:
    winner_indices = [index for index, row in enumerate(rows) if row.winner_present]
    countries = tuple(metadata["countries"])
    country_ids = {country: index for index, country in enumerate(countries)}
    scores: list[float | None] = [None] * len(rows)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(winner_indices), BATCH_SIZE):
            indices = winner_indices[start : start + BATCH_SIZE]
            encoded = [
                encode_utf8(rows[index].normalized_candidate) for index in indices
            ]
            ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
            mask = torch.tensor([item[1] for item in encoded], dtype=torch.bool)
            country = torch.tensor(
                [country_ids.get(rows[index].country_hint, 0) for index in indices],
                dtype=torch.long,
            )
            logits = model(ids, mask, country).tolist()
            for index, logit in zip(indices, logits):
                scores[index] = sigmoid(float(logit))
    return scores


def materialize_rows(
    raw_rows: tuple[RawRow, ...], scores: list[float | None]
) -> tuple[ConditionalRow, ...]:
    if len(raw_rows) != len(scores):
        raise ValueError("conditional scores are not aligned")
    rows = []
    for raw, neural_score in zip(raw_rows, scores):
        if raw.winner_present:
            if neural_score is None:
                raise ValueError("winner is missing a neural score")
            assert raw.candidate_count is not None
            assert raw.role_signal is not None
            assert raw.reliability is not None
            assert raw.winner_margin is not None
            assert raw.candidate_quality is not None
            eligible = (
                raw.c4_source == "abstain" and raw.native_candidate and raw.vetoes_pass
            )
            rows.append(
                ConditionalRow(
                    population=raw.population,
                    expected_greeting=raw.expected_greeting,
                    selected_matches=raw.selected_matches,
                    winner_present=True,
                    c4_emits=raw.c4_emits,
                    c5_emits=raw.c5_emits,
                    candidate_count=raw.candidate_count,
                    eligible=eligible,
                    role_signal=raw.role_signal,
                    reliability=raw.reliability,
                    winner_margin=raw.winner_margin,
                    candidate_quality=raw.candidate_quality,
                    neural_score=neural_score,
                )
            )
        else:
            rows.append(
                ConditionalRow(
                    population=raw.population,
                    expected_greeting=raw.expected_greeting,
                    selected_matches=False,
                    winner_present=False,
                    c4_emits=False,
                    c5_emits=False,
                    candidate_count=0,
                    eligible=False,
                    role_signal=0.0,
                    reliability=0.0,
                    winner_margin=0.0,
                    candidate_quality=0.0,
                    neural_score=0.0,
                )
            )
    return tuple(rows)


def build_analysis(rows: tuple[ConditionalRow, ...]) -> Analysis:
    masks = build_masks(rows)
    baseline, augmented = enumerate_policies(masks)
    baseline_signatures = frozenset(baseline)
    baseline_policies = tuple(baseline.values())
    augmented_policies = tuple(augmented.values())
    baseline_frontier = pareto_frontier(baseline_policies, masks, masks.all_rows)
    augmented_frontier = pareto_frontier(augmented_policies, masks, masks.all_rows)
    same_error = build_same_error_rows(
        baseline_frontier,
        augmented_frontier,
        baseline_signatures,
        masks,
    )
    logo_folds, logo_oof = build_logo_rows(
        baseline_policies,
        augmented_policies,
        baseline_signatures,
        masks,
    )
    recommendation, reason = recommend(same_error, logo_oof)
    conditional_buckets, suppressed_rows, suppressed_strata = build_conditional_buckets(
        rows
    )
    return Analysis(
        population_summary=build_population_summary(rows, masks),
        conditional_buckets=conditional_buckets,
        full_frontiers=build_frontier_rows(
            baseline_frontier, augmented_frontier, baseline_signatures, masks
        ),
        same_error=same_error,
        logo_folds=logo_folds,
        logo_oof=logo_oof,
        recommendation=recommendation,
        recommendation_reason=reason,
        suppressed_bucket_rows=suppressed_rows,
        suppressed_bucket_strata=suppressed_strata,
    )


def build_masks(rows: tuple[ConditionalRow, ...]) -> Masks:
    all_rows = (1 << len(rows)) - 1
    population_masks = {
        population: bitmask(
            index for index, row in enumerate(rows) if row.population == population
        )
        for population in POPULATIONS
    }
    eligible = bitmask(index for index, row in enumerate(rows) if row.eligible)
    quality = threshold_masks(rows, eligible, "candidate_quality", EVIDENCE_GRID)
    reliability = threshold_masks(rows, eligible, "reliability", EVIDENCE_GRID)
    role = threshold_masks(rows, eligible, "role_signal", EVIDENCE_GRID)
    margin = threshold_masks(rows, eligible, "winner_margin", MARGIN_GRID)
    neural = threshold_masks(rows, eligible, "neural_score", (0.0, *NEURAL_GRID))
    if neural[0.0] != eligible:
        raise ValueError("neural N=0 invariant failed")
    return Masks(
        all_rows=all_rows,
        populations=population_masks,
        c4=bitmask(index for index, row in enumerate(rows) if row.c4_emits),
        c5=bitmask(index for index, row in enumerate(rows) if row.c5_emits),
        eligible=eligible,
        correct=bitmask(
            index
            for index, row in enumerate(rows)
            if row.winner_present and row.selected_matches
        ),
        wrong_winner=bitmask(
            index
            for index, row in enumerate(rows)
            if row.winner_present and row.expected_greeting and not row.selected_matches
        ),
        null_winner=bitmask(
            index
            for index, row in enumerate(rows)
            if row.winner_present and not row.expected_greeting
        ),
        sole=bitmask(
            index
            for index, row in enumerate(rows)
            if row.eligible and row.candidate_count == 1
        ),
        multiple=bitmask(
            index
            for index, row in enumerate(rows)
            if row.eligible and row.candidate_count >= 2
        ),
        quality=quality,
        reliability=reliability,
        role=role,
        margin=margin,
        neural=neural,
    )


def enumerate_policies(masks: Masks) -> tuple[dict[int, Policy], dict[int, Policy]]:
    baseline: dict[int, Policy] = {}
    augmented: dict[int, Policy] = {}
    for quality in EVIDENCE_GRID:
        for reliability in EVIDENCE_GRID:
            for role in EVIDENCE_GRID:
                common = masks.quality[quality] & masks.reliability[reliability]
                common &= masks.role[role]
                for margin in MARGIN_GRID:
                    signature = common & (
                        masks.sole | (masks.multiple & masks.margin[margin])
                    )
                    policy = Policy(
                        "baseline",
                        quality,
                        reliability,
                        role,
                        margin,
                        None,
                        signature,
                        signature,
                    )
                    retain_representative(baseline, policy)
                    for neural in NEURAL_GRID:
                        neural_signature = signature & masks.neural[neural]
                        candidate = Policy(
                            "augmented",
                            quality,
                            reliability,
                            role,
                            margin,
                            neural,
                            neural_signature,
                            signature,
                        )
                        retain_representative(augmented, candidate)
    return baseline, augmented


def pareto_frontier(
    policies: Iterable[Policy], masks: Masks, scope: int
) -> list[tuple[Policy, Counts]]:
    best_by_error: dict[int, tuple[Policy, Counts]] = {}
    for policy in policies:
        counts = policy_counts(policy, masks, scope)
        current = best_by_error.get(counts.errors)
        if current is None or evaluated_policy_key(
            policy, counts
        ) > evaluated_policy_key(*current):
            best_by_error[counts.errors] = (policy, counts)
    frontier = []
    best_correct = -1
    for errors in sorted(best_by_error):
        point = best_by_error[errors]
        if point[1].correct > best_correct:
            frontier.append(point)
            best_correct = point[1].correct
    return frontier


def build_same_error_rows(
    baseline_frontier: list[tuple[Policy, Counts]],
    augmented_frontier: list[tuple[Policy, Counts]],
    baseline_signatures: frozenset[int],
    masks: Masks,
) -> list[dict[str, Any]]:
    maximum_budget = max(
        baseline_frontier[-1][1].errors,
        augmented_frontier[-1][1].errors,
    )
    rows = []
    for budget in range(maximum_budget + 1):
        baseline_policy, baseline_counts = best_frontier_at_budget(
            baseline_frontier, budget
        )
        augmented_policy, augmented_counts = best_frontier_at_budget(
            augmented_frontier, budget
        )
        binding = augmented_policy.signature != augmented_policy.parent_signature
        representable = augmented_policy.signature in baseline_signatures
        rows.append(
            {
                "error_budget": budget,
                "highlighted": budget in HIGHLIGHT_ERROR_BUDGETS,
                **comparison_columns(
                    baseline_policy,
                    baseline_counts,
                    augmented_policy,
                    augmented_counts,
                ),
                "correct_gain": augmented_counts.correct - baseline_counts.correct,
                "error_delta": augmented_counts.errors - baseline_counts.errors,
                "nn_binding": binding,
                "nn_signature_baseline_representable": representable,
                "strict_augmented_domination": augmented_counts.correct
                > baseline_counts.correct
                and augmented_counts.errors <= baseline_counts.errors
                and binding
                and not representable,
                **paired_population_columns(
                    baseline_policy, augmented_policy, masks, masks.populations
                ),
            }
        )
    return rows


def build_logo_rows(
    baseline: tuple[Policy, ...],
    augmented: tuple[Policy, ...],
    baseline_signatures: frozenset[int],
    masks: Masks,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds = []
    selected: dict[tuple[int, str, str], Policy] = {}
    for budget in LOGO_ERROR_BUDGETS:
        for heldout in POPULATIONS:
            heldout_scope = masks.populations[heldout]
            training_scope = masks.all_rows & ~heldout_scope
            baseline_policy, baseline_train = select_at_budget(
                baseline, masks, training_scope, budget
            )
            augmented_policy, augmented_train = select_at_budget(
                augmented, masks, training_scope, budget
            )
            baseline_heldout = policy_counts(baseline_policy, masks, heldout_scope)
            augmented_heldout = policy_counts(augmented_policy, masks, heldout_scope)
            selected[(budget, heldout, "baseline")] = baseline_policy
            selected[(budget, heldout, "augmented")] = augmented_policy
            folds.append(
                {
                    "error_budget": budget,
                    "heldout_population": heldout,
                    **comparison_columns(
                        baseline_policy,
                        baseline_train,
                        augmented_policy,
                        augmented_train,
                        prefix="training_",
                    ),
                    **comparison_columns(
                        baseline_policy,
                        baseline_heldout,
                        augmented_policy,
                        augmented_heldout,
                        prefix="heldout_",
                        include_policy=False,
                    ),
                    "heldout_correct_gain": augmented_heldout.correct
                    - baseline_heldout.correct,
                    "heldout_error_delta": augmented_heldout.errors
                    - baseline_heldout.errors,
                    "nn_binding": augmented_policy.signature
                    != augmented_policy.parent_signature,
                    "nn_signature_baseline_representable": augmented_policy.signature
                    in baseline_signatures,
                }
            )

    oof = []
    for budget in LOGO_ERROR_BUDGETS:
        baseline_counts = Counts(0, 0, 0)
        augmented_counts = Counts(0, 0, 0)
        generation_columns: dict[str, Any] = {}
        bindings = []
        representable = []
        no_generation_adds_errors = True
        no_generation_loses_correct = True
        for population in POPULATIONS:
            scope = masks.populations[population]
            baseline_policy = selected[(budget, population, "baseline")]
            augmented_policy = selected[(budget, population, "augmented")]
            baseline_heldout = policy_counts(baseline_policy, masks, scope)
            augmented_heldout = policy_counts(augmented_policy, masks, scope)
            baseline_counts = add_counts(baseline_counts, baseline_heldout)
            augmented_counts = add_counts(augmented_counts, augmented_heldout)
            short = population_short(population)
            generation_columns[f"{short}_baseline_correct"] = baseline_heldout.correct
            generation_columns[f"{short}_baseline_errors"] = baseline_heldout.errors
            generation_columns[f"{short}_augmented_correct"] = augmented_heldout.correct
            generation_columns[f"{short}_augmented_errors"] = augmented_heldout.errors
            generation_columns[f"{short}_correct_gain"] = (
                augmented_heldout.correct - baseline_heldout.correct
            )
            generation_columns[f"{short}_error_delta"] = (
                augmented_heldout.errors - baseline_heldout.errors
            )
            bindings.append(
                augmented_policy.signature != augmented_policy.parent_signature
            )
            representable.append(augmented_policy.signature in baseline_signatures)
            no_generation_adds_errors &= (
                augmented_heldout.errors <= baseline_heldout.errors
            )
            no_generation_loses_correct &= (
                augmented_heldout.correct >= baseline_heldout.correct
            )
        oof.append(
            {
                "error_budget": budget,
                "baseline_correct": baseline_counts.correct,
                "baseline_wrong_winner": baseline_counts.wrong_winner,
                "baseline_null_winner": baseline_counts.null_winner,
                "baseline_errors": baseline_counts.errors,
                "augmented_correct": augmented_counts.correct,
                "augmented_wrong_winner": augmented_counts.wrong_winner,
                "augmented_null_winner": augmented_counts.null_winner,
                "augmented_errors": augmented_counts.errors,
                "correct_gain": augmented_counts.correct - baseline_counts.correct,
                "error_delta": augmented_counts.errors - baseline_counts.errors,
                "all_selected_nn_binding": all(bindings),
                "all_selected_nn_nonrepresentable": not any(representable),
                "no_heldout_generation_adds_errors": no_generation_adds_errors,
                "no_heldout_generation_loses_correct": no_generation_loses_correct,
                **generation_columns,
            }
        )
    return folds, oof


def recommend(
    same_error: list[dict[str, Any]], logo_oof: list[dict[str, Any]]
) -> tuple[str, str]:
    pooled = [row for row in same_error if row["strict_augmented_domination"]]
    promising_oof = [
        row
        for row in logo_oof
        if row["correct_gain"] > 0
        and row["error_delta"] <= 0
        and row["all_selected_nn_binding"]
        and row["all_selected_nn_nonrepresentable"]
        and row["no_heldout_generation_adds_errors"]
    ]
    stable = [
        row for row in promising_oof if row["no_heldout_generation_loses_correct"]
    ]
    if pooled and stable:
        best = max(stable, key=lambda row: (row["correct_gain"], -row["error_budget"]))
        return (
            "proceed to a classifier-integration experiment",
            (
                "The NN is binding and non-representable, strictly improves the pooled "
                f"frontier, and adds {best['correct_gain']} OOF correct emissions without "
                f"additional errors at training budget {best['error_budget']} or in any "
                "held-out generation."
            ),
        )
    if pooled:
        best = max(pooled, key=lambda row: (row["correct_gain"], -row["error_budget"]))
        oof_detail = "No low-error OOF budget supplies a net correct gain."
        if promising_oof:
            best_oof = max(
                promising_oof,
                key=lambda row: (row["correct_gain"], -row["error_budget"]),
            )
            reversed_populations = [
                population_short(population).upper()
                for population in POPULATIONS
                if best_oof[f"{population_short(population)}_correct_gain"] < 0
            ]
            oof_detail = (
                f"At OOF training budget {best_oof['error_budget']}, the combined "
                f"gain is +{best_oof['correct_gain']} correct with "
                f"{-best_oof['error_delta']} fewer errors, but correct admissions "
                f"regress in {', '.join(reversed_populations)}."
            )
        return (
            "conditional signal, not integration-ready",
            (
                "A binding, non-representable pooled gain exists "
                f"(+{best['correct_gain']} correct at error budget {best['error_budget']}), "
                f"but the leave-one-generation-out result is inconsistent. {oof_detail}"
            ),
        )
    return (
        "discard",
        (
            "No binding augmented policy strictly improves the pooled same-error frontier "
            "with a signature unavailable to the handcrafted grid."
        ),
    )


def build_population_summary(
    rows: tuple[ConditionalRow, ...], masks: Masks
) -> list[dict[str, Any]]:
    summaries = []
    scopes = (("COMBINED", masks.all_rows), *masks.populations.items())
    for population, scope in scopes:
        c4 = counts_for_mask(masks.c4, masks, scope)
        c5 = counts_for_mask(masks.c5, masks, scope)
        eligible = masks.eligible & scope
        eligible_counts = counts_for_mask(eligible, masks, scope)
        selected_rows = [
            row
            for row in rows
            if population == "COMBINED" or row.population == population
        ]
        eligible_scores = [row.neural_score for row in selected_rows if row.eligible]
        summaries.append(
            {
                "population": population,
                "rows": len(selected_rows),
                "rows_with_winner": sum(row.winner_present for row in selected_rows),
                "c4_correct": c4.correct,
                "c4_wrong_winner": c4.wrong_winner,
                "c4_null_winner": c4.null_winner,
                "c4_errors": c4.errors,
                "c5_correct": c5.correct,
                "c5_wrong_winner": c5.wrong_winner,
                "c5_null_winner": c5.null_winner,
                "c5_errors": c5.errors,
                "eligible_c4_abstentions": eligible.bit_count(),
                "eligible_correct": eligible_counts.correct,
                "eligible_wrong_winner": eligible_counts.wrong_winner,
                "eligible_null_winner": eligible_counts.null_winner,
                "eligible_nn_mean": sum(eligible_scores) / len(eligible_scores)
                if eligible_scores
                else None,
            }
        )
    return summaries


def build_conditional_buckets(
    rows: tuple[ConditionalRow, ...],
) -> tuple[list[dict[str, Any]], int, int]:
    buckets: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        if not row.eligible:
            continue
        key = (
            "sole" if row.candidate_count == 1 else "multiple",
            grid_bin(row.candidate_quality, EVIDENCE_GRID),
            grid_bin(row.reliability, EVIDENCE_GRID),
            grid_bin(row.role_signal, EVIDENCE_GRID),
            "not_applicable"
            if row.candidate_count == 1
            else grid_bin(row.winner_margin, MARGIN_GRID),
            neural_bin(row.neural_score),
        )
        buckets[key][outcome(row)] += 1
    output = []
    suppressed_rows = 0
    suppressed = 0
    for key in sorted(buckets):
        counts = buckets[key]
        total = counts.total()
        if total < MINIMUM_BUCKET_ROWS:
            suppressed_rows += total
            suppressed += 1
            continue
        correct = counts["correct"]
        errors = counts["wrong_winner"] + counts["null_winner"]
        output.append(
            {
                "population": "COMBINED",
                "candidate_topology": key[0],
                "quality_bin": key[1],
                "reliability_bin": key[2],
                "role_bin": key[3],
                "margin_bin": key[4],
                "neural_bin": key[5],
                "rows": total,
                "correct": correct,
                "wrong_winner": counts["wrong_winner"],
                "null_winner": counts["null_winner"],
                "errors": errors,
                "observed_precision": correct / total,
            }
        )
    return output, suppressed_rows, suppressed


def build_frontier_rows(
    baseline: list[tuple[Policy, Counts]],
    augmented: list[tuple[Policy, Counts]],
    baseline_signatures: frozenset[int],
    masks: Masks,
) -> list[dict[str, Any]]:
    rows = []
    for frontier in (baseline, augmented):
        for policy, counts in frontier:
            row = {
                **policy_columns(policy),
                **count_columns(counts),
                "nn_binding": policy.family == "augmented"
                and policy.signature != policy.parent_signature,
                "signature_baseline_representable": policy.signature
                in baseline_signatures,
            }
            for population, scope in masks.populations.items():
                short = population_short(population)
                population_counts = policy_counts(policy, masks, scope)
                row.update(
                    {
                        f"{short}_correct": population_counts.correct,
                        f"{short}_wrong_winner": population_counts.wrong_winner,
                        f"{short}_null_winner": population_counts.null_winner,
                        f"{short}_errors": population_counts.errors,
                    }
                )
            rows.append(row)
    return rows


def render_analysis(
    analysis: Analysis, selection_receipt: dict[str, Any]
) -> dict[str, bytes]:
    outputs = {
        "population_summary.csv": render_csv(analysis.population_summary),
        "conditional_buckets.csv": render_csv(analysis.conditional_buckets),
        "full_frontiers.csv": render_csv(analysis.full_frontiers),
        "same_error_comparison.csv": render_csv(analysis.same_error),
        "logo_folds.csv": render_csv(analysis.logo_folds),
        "logo_oof_comparison.csv": render_csv(analysis.logo_oof),
    }
    outputs["report.md"] = render_report(analysis, selection_receipt).encode()
    return outputs


def render_report(analysis: Analysis, selection_receipt: dict[str, Any]) -> str:
    highlighted = [row for row in analysis.same_error if row["highlighted"]]
    pooled_gains = [
        row for row in analysis.same_error if row["strict_augmented_domination"]
    ]
    combined = analysis.population_summary[0]
    lines = [
        "# Frozen NN conditional-value experiment",
        "",
        "## Result",
        "",
        f"**Recommendation: {analysis.recommendation}.** {analysis.recommendation_reason}",
        "",
        (
            "This is a diagnostic search over spent REAL_PROXY_V1_DEV, V3, and V4. "
            "It does not define or tune a production emission rule."
        ),
        "",
        "## Frozen inputs and semantics",
        "",
        (
            f"- Model: `{selection_receipt['selected_config']}`, epoch "
            f"{selection_receipt['selected_epoch']}, SHA-256 `{MODEL_SHA256}`."
        ),
        (
            f"- Parameter count: 13,505; morphology TEST receipt SHA-256 "
            f"`{TEST_RECEIPT_SHA256}`. No morphology TEST rows were read."
        ),
        (
            "- Every searched policy is additive over frozen C4 and applies only to "
            "native, veto-free C4 abstentions with a selected winner."
        ),
        (
            "- Baseline gates use candidate quality, reliability, role signal, and "
            "winner margin (margin only for multiple candidates). Augmented gates add "
            "one lower bound on the frozen uncalibrated neural score."
        ),
        "- An error is either a wrong winner or an expected-NULL winner.",
        "",
        (
            "Frozen C4 emits "
            f"{combined['c4_correct']} correct and {combined['c4_errors']} errors. "
            "Frozen production C5 is the no-NN reference point and emits "
            f"{combined['c5_correct']} correct and {combined['c5_errors']} errors in "
            "total, an additional "
            f"{combined['c5_correct'] - combined['c4_correct']} correct and "
            f"{combined['c5_errors'] - combined['c4_errors']} errors over C4."
        ),
        "",
        "## Pooled same-error comparison",
        "",
        "The counts below are additional admissions over the identical frozen C4 base.",
        "",
        (
            "| Error budget | Baseline additional correct/errors | NN additional "
            "correct/errors | Gain | NN binding | Baseline-representable |"
        ),
        "| ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for row in highlighted:
        lines.append(
            f"| {row['error_budget']} | {row['baseline_correct']}/"
            f"{row['baseline_errors']} | {row['augmented_correct']}/"
            f"{row['augmented_errors']} | {row['correct_gain']:+d} | "
            f"{yes_no(row['nn_binding'])} | "
            f"{yes_no(row['nn_signature_baseline_representable'])} |"
        )
    if pooled_gains:
        best_pooled = max(
            pooled_gains,
            key=lambda row: (row["correct_gain"], -row["error_budget"]),
        )
        lines.extend(
            [
                "",
                (
                    "The best strict pooled NN gain is "
                    f"+{best_pooled['correct_gain']} correct at error budget "
                    f"{best_pooled['error_budget']} "
                    f"(`Q={best_pooled['augmented_quality_min']}`, "
                    f"`R={best_pooled['augmented_reliability_min']}`, "
                    f"`L={best_pooled['augmented_role_min']}`, "
                    f"`M={best_pooled['augmented_margin_min']}`, "
                    f"`N={best_pooled['augmented_neural_min']}`)."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "### Complete collapsed budget ranges",
            "",
            (
                "Consecutive budgets with the same paired policies are collapsed. `*` "
                "marks a range containing a highlighted budget."
            ),
            "",
            (
                "| Budget range | Baseline correct/errors | NN correct/errors | Gain | "
                "N | Strict pooled gain |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for start, end, row in collapse_same_error_rows(analysis.same_error):
        budget_range = str(start) if start == end else f"{start}–{end}"
        marker = (
            "*"
            if any(
                budget in HIGHLIGHT_ERROR_BUDGETS for budget in range(start, end + 1)
            )
            else ""
        )
        lines.append(
            f"| {budget_range}{marker} | {row['baseline_correct']}/"
            f"{row['baseline_errors']} | {row['augmented_correct']}/"
            f"{row['augmented_errors']} | {row['correct_gain']:+d} | "
            f"{row['augmented_neural_min']} | "
            f"{yes_no(row['strict_augmented_domination'])} |"
        )
    lines.extend(
        [
            "",
            (
                "The complete integer-budget comparison and both Pareto frontiers are "
                "in `same_error_comparison.csv` and `full_frontiers.csv`."
            ),
            "",
            "## Leave-one-generation-out diagnostic",
            "",
            (
                "Each row selects thresholds using the other two generations at the "
                "fixed training error budget, then combines the untouched held-out "
                "fold predictions. There is no post-OOF threshold selection."
            ),
            "",
            (
                "| Training error budget | Baseline OOF correct/errors | NN OOF "
                "correct/errors | Gain | Error delta | No generation adds errors | "
                "No generation loses correct |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
        ]
    )
    for row in analysis.logo_oof:
        lines.append(
            f"| {row['error_budget']} | {row['baseline_correct']}/"
            f"{row['baseline_errors']} | {row['augmented_correct']}/"
            f"{row['augmented_errors']} | {row['correct_gain']:+d} | "
            f"{row['error_delta']:+d} | "
            f"{yes_no(row['no_heldout_generation_adds_errors'])} | "
            f"{yes_no(row['no_heldout_generation_loses_correct'])} |"
        )
    lines.extend(
        [
            "",
            "## Conditional buckets",
            "",
            (
                "`conditional_buckets.csv` compares NN bands only inside tight bins of "
                "quality, reliability, role signal, margin, and candidate topology. "
                f"Strata with fewer than {MINIMUM_BUCKET_ROWS} rows are omitted; "
                f"{analysis.suppressed_bucket_rows} rows across "
                f"{analysis.suppressed_bucket_strata} strata were suppressed. Outputs "
                "contain counts only—no candidate strings or row identifiers."
            ),
            "",
            "## Scope boundary",
            "",
            (
                "The network was not retrained, quantized, or integrated. C3.1, C4, "
                "C5, the evaluator, holdouts, corpus, and compact artifact were not "
                "changed."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def collapse_same_error_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[int, int, dict[str, Any]]]:
    policy_fields = (
        "baseline_quality_min",
        "baseline_reliability_min",
        "baseline_role_min",
        "baseline_margin_min",
        "augmented_quality_min",
        "augmented_reliability_min",
        "augmented_role_min",
        "augmented_margin_min",
        "augmented_neural_min",
    )
    collapsed = []
    start = 0
    for index in range(1, len(rows) + 1):
        changed = index == len(rows) or any(
            rows[index][field] != rows[start][field] for field in policy_fields
        )
        if changed:
            collapsed.append(
                (
                    rows[start]["error_budget"],
                    rows[index - 1]["error_budget"],
                    rows[start],
                )
            )
            start = index
    return collapsed


def parse_raw_row(record: dict[str, str]) -> RawRow:
    return RawRow(
        population=record["population"],
        normalized_candidate=record["normalized_candidate"],
        country_hint=record["country_hint"],
        expected_greeting=parse_bool(record["expected_greeting"]),
        selected_matches=parse_bool(record["selected_matches"]),
        winner_present=parse_bool(record["winner_present"]),
        c31_emits=parse_bool(record["c31_emits"]),
        c4_emits=parse_bool(record["c4_emits"]),
        c4_source=record["c4_source"],
        c5_emits=parse_bool(record["c5_emits"]),
        candidate_count=parse_optional_int(record["candidate_count"]),
        native_candidate=parse_bool(record["native_candidate"]),
        segmented_candidate=parse_optional_bool(record["segmented_candidate"]),
        vetoes_pass=parse_bool(record["vetoes_pass"]),
        hard_organization_marker=parse_bool(record["hard_organization_marker"]),
        generic_organization_marker=parse_bool(record["generic_organization_marker"]),
        ampersand=parse_bool(record["ampersand"]),
        candidate_too_short=parse_bool(record["candidate_too_short"]),
        role_signal=parse_optional_float(record["role_signal"]),
        reliability=parse_optional_float(record["reliability"]),
        winner_margin=parse_optional_float(record["winner_margin"]),
        candidate_quality=parse_optional_float(record["candidate_quality"]),
    )


def validate_raw_row(row: RawRow) -> None:
    vetoes_pass = not (
        row.hard_organization_marker
        or row.generic_organization_marker
        or row.ampersand
        or row.candidate_too_short
    )
    if row.vetoes_pass != vetoes_pass:
        raise ValueError("conditional veto fields disagree")
    if row.c4_source not in {"c3_1", "sole_native", "dominant_winner", "abstain"}:
        raise ValueError(f"unexpected C4 source: {row.c4_source}")
    if row.c31_emits and (not row.c4_emits or row.c4_source != "c3_1"):
        raise ValueError("C3.1 emission is not preserved by C4")
    if row.c4_emits != (row.c4_source != "abstain"):
        raise ValueError("C4 source and emission disagree")
    if row.c4_emits and not row.c5_emits:
        raise ValueError("C5 does not preserve a C4 emission")
    if row.native_candidate != (row.segmented_candidate is False):
        raise ValueError("native and segmented flags disagree")
    optional = (
        row.candidate_count,
        row.role_signal,
        row.reliability,
        row.winner_margin,
        row.candidate_quality,
    )
    if row.winner_present:
        if not row.normalized_candidate or any(value is None for value in optional):
            raise ValueError("winner row is missing candidate evidence")
        if row.candidate_count is not None and row.candidate_count < 1:
            raise ValueError("winner candidate count is below one")
        for value in optional[1:]:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError("winner evidence is outside [0, 1]")
    elif row.normalized_candidate or any(value is not None for value in optional):
        raise ValueError("no-winner row contains candidate evidence")
    elif row.c31_emits or row.c4_emits or row.c5_emits or row.selected_matches:
        raise ValueError("no-winner row contains an emission or match")
    if row.selected_matches and not row.expected_greeting:
        raise ValueError("selected match has no expected greeting")


def threshold_masks(
    rows: tuple[ConditionalRow, ...],
    eligible: int,
    attribute: str,
    thresholds: Iterable[float],
) -> dict[float, int]:
    return {
        threshold: bitmask(
            index
            for index, row in enumerate(rows)
            if eligible & (1 << index) and getattr(row, attribute) >= threshold
        )
        for threshold in thresholds
    }


def retain_representative(target: dict[int, Policy], candidate: Policy) -> None:
    current = target.get(candidate.signature)
    if current is None or policy_key(candidate) > policy_key(current):
        target[candidate.signature] = candidate


def select_at_budget(
    policies: Iterable[Policy], masks: Masks, scope: int, budget: int
) -> tuple[Policy, Counts]:
    selected: tuple[Policy, Counts] | None = None
    for policy in policies:
        counts = policy_counts(policy, masks, scope)
        if counts.errors > budget:
            continue
        if selected is None or evaluated_policy_key(
            policy, counts
        ) > evaluated_policy_key(*selected):
            selected = (policy, counts)
    if selected is None:
        raise ValueError("policy grid has no feasible empty policy")
    return selected


def best_frontier_at_budget(
    frontier: list[tuple[Policy, Counts]], budget: int
) -> tuple[Policy, Counts]:
    feasible = [point for point in frontier if point[1].errors <= budget]
    if not feasible:
        raise ValueError("frontier has no zero-error point")
    return max(feasible, key=lambda point: evaluated_policy_key(*point))


def evaluated_policy_key(policy: Policy, counts: Counts) -> tuple[Any, ...]:
    return (counts.correct, -counts.errors, -counts.emitted, policy_key(policy))


def policy_key(policy: Policy) -> tuple[float, ...]:
    return (
        policy.quality,
        policy.reliability,
        policy.role,
        policy.margin,
        policy.neural if policy.neural is not None else -1.0,
    )


def policy_counts(policy: Policy, masks: Masks, scope: int) -> Counts:
    return counts_for_mask(policy.signature, masks, scope)


def counts_for_mask(signature: int, masks: Masks, scope: int) -> Counts:
    selected = signature & scope
    return Counts(
        correct=(selected & masks.correct).bit_count(),
        wrong_winner=(selected & masks.wrong_winner).bit_count(),
        null_winner=(selected & masks.null_winner).bit_count(),
    )


def add_counts(left: Counts, right: Counts) -> Counts:
    return Counts(
        left.correct + right.correct,
        left.wrong_winner + right.wrong_winner,
        left.null_winner + right.null_winner,
    )


def comparison_columns(
    baseline_policy: Policy,
    baseline_counts: Counts,
    augmented_policy: Policy,
    augmented_counts: Counts,
    *,
    prefix: str = "",
    include_policy: bool = True,
) -> dict[str, Any]:
    result = {
        f"{prefix}baseline_correct": baseline_counts.correct,
        f"{prefix}baseline_wrong_winner": baseline_counts.wrong_winner,
        f"{prefix}baseline_null_winner": baseline_counts.null_winner,
        f"{prefix}baseline_errors": baseline_counts.errors,
        f"{prefix}augmented_correct": augmented_counts.correct,
        f"{prefix}augmented_wrong_winner": augmented_counts.wrong_winner,
        f"{prefix}augmented_null_winner": augmented_counts.null_winner,
        f"{prefix}augmented_errors": augmented_counts.errors,
    }
    if include_policy:
        result.update(prefixed_policy_columns("baseline", baseline_policy, prefix))
        result.update(prefixed_policy_columns("augmented", augmented_policy, prefix))
    return result


def paired_population_columns(
    baseline: Policy,
    augmented: Policy,
    masks: Masks,
    populations: dict[str, int],
) -> dict[str, Any]:
    result = {}
    for population, scope in populations.items():
        short = population_short(population)
        baseline_counts = policy_counts(baseline, masks, scope)
        augmented_counts = policy_counts(augmented, masks, scope)
        result.update(
            {
                f"{short}_baseline_correct": baseline_counts.correct,
                f"{short}_baseline_errors": baseline_counts.errors,
                f"{short}_augmented_correct": augmented_counts.correct,
                f"{short}_augmented_errors": augmented_counts.errors,
            }
        )
    return result


def policy_columns(policy: Policy) -> dict[str, Any]:
    return {
        "family": policy.family,
        "quality_min": policy.quality,
        "reliability_min": policy.reliability,
        "role_min": policy.role,
        "multiple_candidate_margin_min": policy.margin,
        "neural_min": policy.neural,
    }


def prefixed_policy_columns(
    label: str, policy: Policy, prefix: str = ""
) -> dict[str, Any]:
    return {
        f"{prefix}{label}_quality_min": policy.quality,
        f"{prefix}{label}_reliability_min": policy.reliability,
        f"{prefix}{label}_role_min": policy.role,
        f"{prefix}{label}_margin_min": policy.margin,
        f"{prefix}{label}_neural_min": policy.neural,
    }


def count_columns(counts: Counts) -> dict[str, int]:
    return {
        "correct": counts.correct,
        "wrong_winner": counts.wrong_winner,
        "null_winner": counts.null_winner,
        "errors": counts.errors,
        "emitted": counts.emitted,
    }


def bitmask(indices: Iterable[int]) -> int:
    value = 0
    for index in indices:
        value |= 1 << index
    return value


def outcome(row: ConditionalRow) -> str:
    if row.selected_matches:
        return "correct"
    if row.expected_greeting:
        return "wrong_winner"
    return "null_winner"


def grid_bin(value: float, grid: tuple[float, ...]) -> str:
    lower = grid[0]
    for upper in grid[1:]:
        if value < upper:
            return f"[{lower},{upper})"
        lower = upper
    return f"[{grid[-1]}]"


def neural_bin(value: float) -> str:
    if value < 0.25:
        return "[0.0,0.25)"
    if value < 0.5:
        return "[0.25,0.5)"
    if value < 0.75:
        return "[0.5,0.75)"
    if value < 0.9:
        return "[0.75,0.9)"
    if value < 0.975:
        return "[0.9,0.975)"
    return "[0.975,1.0]"


def population_short(population: str) -> str:
    return {
        "REAL_PROXY_V1_DEV": "v1",
        "REAL_PROXY_V3": "v3",
        "REAL_PROXY_V4": "v4",
    }[population]


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def parse_optional_bool(value: str) -> bool | None:
    return parse_bool(value) if value else None


def parse_optional_int(value: str) -> int | None:
    return int(value) if value else None


def parse_optional_float(value: str) -> float | None:
    return float(value) if value else None


def render_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot render an empty CSV")
    destination = io.StringIO(newline="")
    writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return destination.getvalue().encode()


def render_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def yes_no(value: bool) -> str:
    return "yes" if value else "no"
