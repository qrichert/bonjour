from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .data import (
    Example,
    TensorData,
    country_vocabulary,
    encode_utf8,
    load_examples,
    load_json,
    sha256_file,
    tensorize,
    verify_dataset_file,
)
from .metrics import (
    BinaryMetrics,
    average_precision,
    binary_metrics,
    constrained_threshold,
    pearson,
    percentiles,
    roc_auc,
    sigmoid,
    spearman,
)
from .model import (
    CONFIGURATIONS,
    ModelConfig,
    MorphologyModel,
    copy_state,
    export_model,
    load_export,
    parameter_count,
)

SEED = 0x4E4E5F4D4F525048
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
BATCH_SIZE = 512
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 3
COUNTRY_DROPOUT = 0.20
NEGATIVE_POPULATIONS = (
    "strong_surname",
    "surname_only",
    "organization_non_name",
    "garbage_malformed",
    "dictionary_non_name",
)
OUTCOMES = (
    "correct_winner_c31_emits",
    "correct_winner_c31_abstains",
    "wrong_winner",
    "expected_null_winner",
)


@dataclass
class EpochResult:
    config: str
    epoch: int
    rows: int
    roc_auc: float
    average_precision: float
    threshold_fpr_0_001: float
    recall_fpr_0_001: float
    false_positives_fpr_0_001: int
    worst_negative_specificity: float


@dataclass
class TrainedConfiguration:
    config: ModelConfig
    epoch: int
    metrics: BinaryMetrics
    threshold: float
    worst_specificity: float
    parameters: int
    unknown_country_metrics: BinaryMetrics | None
    state: dict[str, torch.Tensor]
    epochs: list[EpochResult]


@dataclass(frozen=True)
class ProxyScoredRow:
    population: str
    outcome: str
    logit: float
    score: float
    role_llr: float
    role_signal: float
    reliability: float
    winner_margin: float
    candidate_quality: float


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        arguments.action(arguments)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    select = commands.add_parser("select")
    select.add_argument("--train", type=Path, required=True)
    select.add_argument("--validation", type=Path, required=True)
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(action=run_select)

    evaluate_test = commands.add_parser("evaluate-test")
    evaluate_test.add_argument("--test", type=Path, required=True)
    evaluate_test.add_argument("--manifest", type=Path, required=True)
    evaluate_test.add_argument("--selection", type=Path, required=True)
    evaluate_test.add_argument("--output", type=Path, required=True)
    evaluate_test.set_defaults(action=run_evaluate_test)

    evaluate_proxy = commands.add_parser("evaluate-proxy")
    evaluate_proxy.add_argument("--selection", type=Path, required=True)
    evaluate_proxy.add_argument("--test", type=Path, required=True)
    evaluate_proxy.add_argument("--output", type=Path, required=True)
    evaluate_proxy.set_defaults(action=run_evaluate_proxy)

    report = commands.add_parser("report")
    report.add_argument("--data", type=Path, required=True)
    report.add_argument("--selection", type=Path, required=True)
    report.add_argument("--test", type=Path, required=True)
    report.add_argument("--proxy", type=Path, required=True)
    report.add_argument("--probes", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(action=run_report)
    return parser


def run_select(arguments: argparse.Namespace) -> None:
    configure_torch()
    manifest = load_json(arguments.manifest)
    verify_dataset_file(arguments.train, manifest)
    verify_dataset_file(arguments.validation, manifest)
    train = load_examples(arguments.train)
    validation = load_examples(arguments.validation)
    require_split(train, "train")
    require_split(validation, "validation")
    require_disjoint(train, validation)
    countries = country_vocabulary(train)
    train_data = tensorize(train, countries)
    validation_data = tensorize(validation, countries)

    trained = [
        train_configuration(config, train_data, validation_data, countries)
        for config in CONFIGURATIONS
    ]
    selected = max(trained, key=configuration_key)
    model = MorphologyModel(selected.config, len(countries))
    model.load_state_dict(selected.state)
    model.eval()
    validation_logits = score_tensor_data(model, validation_data)
    validation_scores = [sigmoid(value) for value in validation_logits]
    thresholds = select_thresholds(
        [example.label for example in validation], validation_scores
    )

    def write(temporary: Path) -> None:
        write_epoch_metrics(temporary / "epoch_metrics.csv", trained)
        write_config_metrics(temporary / "config_metrics.csv", trained)
        write_thresholds(temporary / "frozen_thresholds.csv", thresholds)
        metadata = {
            "countries": list(countries),
            "country_unknown_index": 0,
            "selected_epoch": selected.epoch,
            "seed": SEED,
            "normalization": "canonicalize + Unicode casefold + NFC + UTF-8 bytes",
            "maximum_ids": 96,
            "train_sha256": sha256_file(arguments.train),
            "validation_sha256": sha256_file(arguments.validation),
            "dataset_manifest_sha256": sha256_file(arguments.manifest),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        export_model(
            model,
            metadata,
            temporary / "selected_model.json",
            temporary / "selected_model.f32.bin",
        )
        reloaded, _ = load_export(temporary / "selected_model.f32.bin")
        replay = score_tensor_data(reloaded, validation_data, limit=256)
        maximum_delta = max(
            (abs(left - right) for left, right in zip(validation_logits[:256], replay)),
            default=0.0,
        )
        if maximum_delta > 1.0e-6:
            raise ValueError(f"float32 export parity failed: {maximum_delta}")
        receipt = {
            "format_version": 1,
            "selected_config": selected.config.name,
            "selected_epoch": selected.epoch,
            "model_sha256": sha256_file(temporary / "selected_model.f32.bin"),
            "model_metadata_sha256": sha256_file(temporary / "selected_model.json"),
            "thresholds_sha256": sha256_file(temporary / "frozen_thresholds.csv"),
            "dataset_manifest_sha256": sha256_file(arguments.manifest),
            "train_sha256": sha256_file(arguments.train),
            "validation_sha256": sha256_file(arguments.validation),
            "export_maximum_logit_delta": maximum_delta,
        }
        write_json(temporary / "selection_receipt.json", receipt)

    publish_directory(arguments.output, write)


def run_evaluate_test(arguments: argparse.Namespace) -> None:
    configure_torch()
    manifest = load_json(arguments.manifest)
    verify_dataset_file(arguments.test, manifest)
    selection = verify_selection(arguments.selection)
    if selection["dataset_manifest_sha256"] != sha256_file(arguments.manifest):
        raise ValueError("selection is bound to a different dataset manifest")
    examples = load_examples(arguments.test)
    require_split(examples, "morph_test")
    model, metadata = load_export(arguments.selection / "selected_model.f32.bin")
    data = tensorize(examples, tuple(metadata["countries"]))
    logits = score_tensor_data(model, data)
    scores = [sigmoid(value) for value in logits]
    unknown_logits = score_tensor_data(model, data, force_unknown_country=True)
    unknown_scores = [sigmoid(value) for value in unknown_logits]
    thresholds = read_thresholds(arguments.selection / "frozen_thresholds.csv")

    def write(temporary: Path) -> None:
        write_morphology_metrics(temporary / "morphology_metrics.csv", examples, scores)
        write_country_ablation(
            temporary / "country_ablation.csv",
            examples,
            scores,
            unknown_scores,
            thresholds["fpr_0_001"],
        )
        write_population_percentiles(
            temporary / "population_percentiles.csv", examples, logits, scores
        )
        write_operating_points(
            temporary / "operating_points.csv", examples, scores, thresholds
        )
        write_signal_auc(temporary / "signal_auc.csv", examples, logits)
        write_signal_correlations(
            temporary / "signal_correlations.csv", examples, logits
        )
        write_conditional_slices(
            temporary / "conditional_slices.csv",
            temporary / "local_case_review.csv",
            examples,
            logits,
            scores,
            thresholds["fpr_0_001"],
        )
        summary = test_summary(
            examples,
            logits,
            scores,
            unknown_logits,
            unknown_scores,
            thresholds,
        )
        write_json(temporary / "test_summary.json", summary)
        receipt = {
            "format_version": 1,
            "model_sha256": selection["model_sha256"],
            "thresholds_sha256": selection["thresholds_sha256"],
            "morph_test_sha256": sha256_file(arguments.test),
            "dataset_manifest_sha256": sha256_file(arguments.manifest),
            "test_summary_sha256": sha256_file(temporary / "test_summary.json"),
        }
        write_json(temporary / "test_receipt.json", receipt)

    publish_directory(arguments.output, write)


def run_evaluate_proxy(arguments: argparse.Namespace) -> None:
    configure_torch()
    selection = verify_selection(arguments.selection)
    test_receipt = load_json(arguments.test / "test_receipt.json")
    if test_receipt["model_sha256"] != selection["model_sha256"]:
        raise ValueError("test receipt is bound to a different model")
    if test_receipt["thresholds_sha256"] != selection["thresholds_sha256"]:
        raise ValueError("test receipt is bound to different thresholds")
    model, metadata = load_export(arguments.selection / "selected_model.f32.bin")
    thresholds = read_thresholds(arguments.selection / "frozen_thresholds.csv")
    proxy_rows, no_winner_counts = read_and_score_proxy(
        sys.stdin, model, tuple(metadata["countries"])
    )

    def write(temporary: Path) -> None:
        write_proxy_distributions(
            temporary / "proxy_distributions.csv", proxy_rows, no_winner_counts
        )
        write_proxy_2d(temporary / "proxy_2d_role_neural.csv", proxy_rows)
        write_proxy_correlations(temporary / "proxy_correlations.csv", proxy_rows)
        write_proxy_thresholds(
            temporary / "proxy_threshold_diagnostics.csv", proxy_rows, thresholds
        )
        summary = proxy_summary(proxy_rows, no_winner_counts, thresholds["fpr_0_001"])
        write_json(temporary / "proxy_summary.json", summary)
        receipt = {
            "format_version": 1,
            "model_sha256": selection["model_sha256"],
            "thresholds_sha256": selection["thresholds_sha256"],
            "test_receipt_sha256": sha256_file(arguments.test / "test_receipt.json"),
            "accepted_proxy_digests": {
                "REAL_PROXY_V1_DEV": "de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e",
                "REAL_PROXY_V3": "d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe",
                "REAL_PROXY_V4": "d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f",
            },
            "rows_with_winner": len(proxy_rows),
            "rows_without_winner": sum(no_winner_counts.values()),
        }
        write_json(temporary / "evaluation_receipt.json", receipt)

    publish_directory(arguments.output, write)


def run_report(arguments: argparse.Namespace) -> None:
    from .report import build_report

    build_report(
        data=arguments.data,
        selection=arguments.selection,
        test=arguments.test,
        proxy=arguments.proxy,
        probes=arguments.probes,
        output=arguments.output,
    )


def configure_torch() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)


def train_configuration(
    config: ModelConfig,
    train: TensorData,
    validation: TensorData,
    countries: tuple[str, ...],
) -> TrainedConfiguration:
    torch.manual_seed(SEED)
    model = MorphologyModel(config, len(countries))
    if parameter_count(model) >= 500_000:
        raise ValueError(f"{config.name} exceeds the parameter ceiling")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    positive = int(train.labels.sum().item())
    negative = len(train.examples) - positive
    if positive == 0 or negative == 0:
        raise ValueError("training requires both labels")
    class_weights = torch.tensor(
        [len(train.examples) / (2 * negative), len(train.examples) / (2 * positive)],
        dtype=torch.float32,
    )
    epochs: list[EpochResult] = []
    best: TrainedConfiguration | None = None
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, optimizer, train, class_weights, config, epoch)
        logits = score_tensor_data(model, validation)
        scores = [sigmoid(value) for value in logits]
        labels = [example.label for example in validation.examples]
        threshold, metrics = constrained_threshold(labels, scores, maximum_fpr=0.001)
        worst = worst_population_specificity(validation.examples, scores, threshold)
        epoch_result = EpochResult(
            config=config.name,
            epoch=epoch,
            rows=len(labels),
            roc_auc=metrics.roc_auc,
            average_precision=metrics.average_precision,
            threshold_fpr_0_001=threshold,
            recall_fpr_0_001=metrics.recall,
            false_positives_fpr_0_001=metrics.false_positives,
            worst_negative_specificity=worst,
        )
        epochs.append(epoch_result)
        candidate = TrainedConfiguration(
            config=config,
            epoch=epoch,
            metrics=metrics,
            threshold=threshold,
            worst_specificity=worst,
            parameters=parameter_count(model),
            unknown_country_metrics=None,
            state=copy_state(model),
            epochs=[],
        )
        if best is None or checkpoint_key(candidate) > checkpoint_key(best):
            best = candidate
            stale = 0
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                break
    assert best is not None
    model.load_state_dict(best.state)
    if config.country_embedding:
        unknown_logits = score_tensor_data(
            model, validation, force_unknown_country=True
        )
        unknown_scores = [sigmoid(value) for value in unknown_logits]
        best.unknown_country_metrics = binary_metrics(
            [example.label for example in validation.examples],
            unknown_scores,
            best.threshold,
        )
    best.epochs = epochs
    return best


def train_epoch(
    model: MorphologyModel,
    optimizer: torch.optim.Optimizer,
    data: TensorData,
    class_weights: torch.Tensor,
    config: ModelConfig,
    epoch: int,
) -> None:
    model.train()
    generator = torch.Generator().manual_seed(SEED + epoch)
    order = torch.randperm(len(data.examples), generator=generator)
    for start in range(0, len(data.examples), BATCH_SIZE):
        indices = order[start : start + BATCH_SIZE]
        ids = data.ids.index_select(0, indices)
        mask = data.mask.index_select(0, indices)
        countries = data.countries.index_select(0, indices)
        labels = data.labels.index_select(0, indices)
        if config.country_embedding:
            dropout = torch.rand(countries.shape, generator=generator) < COUNTRY_DROPOUT
            countries = countries.masked_fill(dropout, 0)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, mask, countries)
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = (losses * class_weights[labels.to(torch.long)]).mean()
        loss.backward()
        optimizer.step()


def score_tensor_data(
    model: MorphologyModel,
    data: TensorData,
    *,
    limit: int | None = None,
    force_unknown_country: bool = False,
) -> list[float]:
    model.eval()
    size = min(len(data.examples), limit) if limit is not None else len(data.examples)
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, size, BATCH_SIZE):
            end = min(size, start + BATCH_SIZE)
            countries = data.countries[start:end]
            if force_unknown_country:
                countries = torch.zeros_like(countries)
            logits = model(data.ids[start:end], data.mask[start:end], countries)
            scores.extend(float(value) for value in logits.tolist())
    return scores


def checkpoint_key(
    value: TrainedConfiguration,
) -> tuple[float, float, float, float, int]:
    return (
        value.metrics.recall,
        value.worst_specificity,
        value.metrics.average_precision,
        value.metrics.roc_auc,
        -value.epoch,
    )


def configuration_key(
    value: TrainedConfiguration,
) -> tuple[float, float, float, float, int, str]:
    return (*checkpoint_key(value)[:4], -value.parameters, value.config.name)


def worst_population_specificity(
    examples: tuple[Example, ...], scores: list[float], threshold: float
) -> float:
    values = []
    for population in NEGATIVE_POPULATIONS:
        selected = [
            score
            for example, score in zip(examples, scores)
            if population in example.populations
        ]
        if selected:
            values.append(sum(score < threshold for score in selected) / len(selected))
    return min(values) if values else math.nan


def select_thresholds(labels: list[int], scores: list[float]) -> dict[str, float]:
    thresholds = {"fixed_0_5": 0.5}
    for name, maximum in (
        ("zero_false_positives", 0.0),
        ("fpr_0_001", 0.001),
        ("fpr_0_005", 0.005),
        ("fpr_0_01", 0.01),
    ):
        thresholds[name] = constrained_threshold(labels, scores, maximum_fpr=maximum)[0]
    for name, minimum in (("precision_0_99", 0.99), ("precision_0_995", 0.995)):
        thresholds[name] = constrained_threshold(
            labels, scores, minimum_precision=minimum
        )[0]
    return thresholds


def write_epoch_metrics(path: Path, trained: list[TrainedConfiguration]) -> None:
    write_csv(path, [asdict(epoch) for value in trained for epoch in value.epochs])


def write_config_metrics(path: Path, trained: list[TrainedConfiguration]) -> None:
    rows = []
    for value in trained:
        unknown = value.unknown_country_metrics
        rows.append(
            {
                "config": value.config.name,
                "selected_epoch": value.epoch,
                "parameters": value.parameters,
                "validation_roc_auc": value.metrics.roc_auc,
                "validation_average_precision": value.metrics.average_precision,
                "validation_threshold_fpr_0_001": value.threshold,
                "validation_recall_fpr_0_001": value.metrics.recall,
                "validation_false_positives_fpr_0_001": value.metrics.false_positives,
                "validation_worst_negative_specificity": value.worst_specificity,
                "forced_unknown_roc_auc": unknown.roc_auc if unknown else math.nan,
                "forced_unknown_average_precision": unknown.average_precision
                if unknown
                else math.nan,
                "forced_unknown_recall_at_recorded_threshold": unknown.recall
                if unknown
                else math.nan,
            }
        )
    write_csv(path, rows)


def write_thresholds(path: Path, thresholds: dict[str, float]) -> None:
    write_csv(
        path,
        [
            {"name": name, "threshold": threshold}
            for name, threshold in thresholds.items()
        ],
    )


def write_morphology_metrics(
    path: Path, examples: tuple[Example, ...], scores: list[float]
) -> None:
    rows = []
    labels = [example.label for example in examples]
    rows.append(auc_row("pooled", labels, scores))
    for population in NEGATIVE_POPULATIONS:
        selected = [
            (example.label, score)
            for example, score in zip(examples, scores)
            if example.label == 1 or population in example.populations
        ]
        rows.append(
            auc_row(
                f"given_vs_{population}",
                [row[0] for row in selected],
                [row[1] for row in selected],
                status="ok" if any(not row[0] for row in selected) else "unavailable",
            )
        )
    write_csv(path, rows)


def write_country_ablation(
    path: Path,
    examples: tuple[Example, ...],
    scores: list[float],
    unknown_scores: list[float],
    threshold: float,
) -> None:
    labels = [example.label for example in examples]
    rows = []
    for scope, values in (
        ("recorded_country", scores),
        ("forced_unknown_country", unknown_scores),
    ):
        metrics = binary_metrics(labels, values, threshold)
        rows.append(
            {
                "scope": scope,
                "rows": metrics.rows,
                "roc_auc": metrics.roc_auc,
                "average_precision": metrics.average_precision,
                "threshold": threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "false_positives": metrics.false_positives,
                "false_positive_rate": metrics.false_positive_rate,
            }
        )
    write_csv(path, rows)


def write_population_percentiles(
    path: Path,
    examples: tuple[Example, ...],
    logits: list[float],
    scores: list[float],
) -> None:
    populations = ("high_confidence_given",) + NEGATIVE_POPULATIONS
    rows = []
    for population in populations:
        indices = [
            index
            for index, example in enumerate(examples)
            if population in example.populations
        ]
        for feature, values in (
            ("morphology_logit", [logits[index] for index in indices]),
            ("morphology_score_uncalibrated", [scores[index] for index in indices]),
        ):
            rows.append(
                {
                    "population": population,
                    "status": "ok" if indices else "unavailable",
                    "feature": feature,
                    "rows": len(indices),
                    **percentiles(values),
                }
            )
    write_csv(path, rows)


def write_operating_points(
    path: Path,
    examples: tuple[Example, ...],
    scores: list[float],
    thresholds: dict[str, float],
) -> None:
    rows = []
    labels = [example.label for example in examples]
    for name, threshold in thresholds.items():
        rows.append(
            metrics_row(
                "pooled", name, threshold, binary_metrics(labels, scores, threshold)
            )
        )
        for population in NEGATIVE_POPULATIONS:
            selected = [
                (example.label, score)
                for example, score in zip(examples, scores)
                if example.label == 1 or population in example.populations
            ]
            if not selected or not any(label == 0 for label, _ in selected):
                rows.append(
                    unavailable_operating_row(f"given_vs_{population}", name, threshold)
                )
                continue
            metrics = binary_metrics(
                [row[0] for row in selected], [row[1] for row in selected], threshold
            )
            rows.append(metrics_row(f"given_vs_{population}", name, threshold, metrics))
    write_csv(path, rows)


def write_signal_auc(
    path: Path, examples: tuple[Example, ...], logits: list[float]
) -> None:
    signals: tuple[tuple[str, Callable[[Example], float | None]], ...] = (
        ("neural_logit", lambda _example: None),
        ("aggregate_role_llr", lambda example: example.role_llr),
        ("aggregate_role_signal", lambda example: example.role_signal),
        ("log1p_given_count", lambda example: math.log1p(example.given_count)),
        (
            "log1p_total_count",
            lambda example: math.log1p(example.given_count + example.surname_count),
        ),
        (
            "production_candidate_quality",
            lambda example: example.production_candidate_quality,
        ),
        ("production_reliability", lambda example: example.production_reliability),
    )
    rows = []
    for name, accessor in signals:
        selected = []
        for index, example in enumerate(examples):
            value = logits[index] if name == "neural_logit" else accessor(example)
            if value is not None:
                selected.append((example.label, value))
        rows.append(
            auc_row(name, [row[0] for row in selected], [row[1] for row in selected])
        )
    write_csv(path, rows)


def write_signal_correlations(
    path: Path, examples: tuple[Example, ...], logits: list[float]
) -> None:
    signals: tuple[tuple[str, Callable[[Example], float | None]], ...] = (
        ("aggregate_role_llr", lambda example: example.role_llr),
        ("aggregate_role_signal", lambda example: example.role_signal),
        ("log1p_given_count", lambda example: math.log1p(example.given_count)),
        (
            "production_candidate_quality",
            lambda example: example.production_candidate_quality,
        ),
        ("production_reliability", lambda example: example.production_reliability),
    )
    scopes = ("pooled", "high_confidence_given") + NEGATIVE_POPULATIONS
    rows = []
    for scope in scopes:
        indices = [
            index
            for index, example in enumerate(examples)
            if scope == "pooled" or scope in example.populations
        ]
        for name, accessor in signals:
            pairs = [
                (logits[index], accessor(examples[index]))
                for index in indices
                if accessor(examples[index]) is not None
            ]
            left = [pair[0] for pair in pairs]
            right = [float(pair[1]) for pair in pairs]
            rows.append(
                {
                    "scope": scope,
                    "signal": name,
                    "rows": len(pairs),
                    "pearson": pearson(left, right),
                    "spearman": spearman(left, right),
                }
            )
    write_csv(path, rows)


def write_conditional_slices(
    summary_path: Path,
    review_path: Path,
    examples: tuple[Example, ...],
    logits: list[float],
    scores: list[float],
    threshold: float,
) -> None:
    definitions = (
        (
            "role_signal_below_0_4_neural_high",
            lambda example, score: example.role_signal < 0.4 and score >= threshold,
        ),
        (
            "role_signal_0_4_to_0_8_neural_high",
            lambda example, score: (
                0.4 <= example.role_signal < 0.8 and score >= threshold
            ),
        ),
        (
            "neural_false_positive",
            lambda example, score: example.label == 0 and score >= threshold,
        ),
    )
    summaries = []
    reviews = []
    for name, includes in definitions:
        selected = [
            (example, logit, score)
            for example, logit, score in zip(examples, logits, scores)
            if includes(example, score)
        ]
        summaries.append(
            {
                "slice": name,
                "threshold": threshold,
                "rows": len(selected),
                "positives": sum(example.label for example, _, _ in selected),
                "negatives": sum(not example.label for example, _, _ in selected),
            }
        )
        for example, logit, score in sorted(
            selected, key=lambda row: row[2], reverse=True
        )[:20]:
            reviews.append(
                {
                    "slice": name,
                    "normalized": example.normalized,
                    "label": example.label,
                    "populations": ";".join(example.populations),
                    "role_signal": example.role_signal,
                    "morphology_logit": logit,
                    "morphology_score_uncalibrated": score,
                }
            )
    write_csv(summary_path, summaries)
    write_csv(
        review_path,
        reviews,
        fieldnames=(
            "slice",
            "normalized",
            "label",
            "populations",
            "role_signal",
            "morphology_logit",
            "morphology_score_uncalibrated",
        ),
    )


def test_summary(
    examples: tuple[Example, ...],
    logits: list[float],
    scores: list[float],
    unknown_logits: list[float],
    unknown_scores: list[float],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    threshold = thresholds["fpr_0_001"]
    labels = [example.label for example in examples]
    pooled = binary_metrics(labels, scores, threshold)
    unknown = binary_metrics(labels, unknown_scores, threshold)
    negative_fprs = {}
    for population in NEGATIVE_POPULATIONS:
        selected = [
            score
            for example, score in zip(examples, scores)
            if population in example.populations
        ]
        negative_fprs[population] = (
            sum(score >= threshold for score in selected) / len(selected)
            if selected
            else None
        )
    return {
        "rows": len(examples),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "roc_auc": pooled.roc_auc,
        "average_precision": pooled.average_precision,
        "fpr_0_001_threshold": threshold,
        "recall_at_fpr_0_001_threshold": pooled.recall,
        "false_positives_at_fpr_0_001_threshold": pooled.false_positives,
        "observed_fpr_at_fpr_0_001_threshold": pooled.false_positive_rate,
        "negative_population_fpr": negative_fprs,
        "neural_role_spearman": spearman(
            logits, [example.role_signal for example in examples]
        ),
        "forced_unknown_country": {
            "roc_auc": unknown.roc_auc,
            "average_precision": unknown.average_precision,
            "recall_at_fpr_0_001_threshold": unknown.recall,
            "false_positives_at_fpr_0_001_threshold": unknown.false_positives,
            "observed_fpr_at_fpr_0_001_threshold": unknown.false_positive_rate,
            "neural_role_spearman": spearman(
                unknown_logits,
                [example.role_signal for example in examples],
            ),
        },
    }


def read_and_score_proxy(
    source,
    model: MorphologyModel,
    countries: tuple[str, ...],
) -> tuple[list[ProxyScoredRow], dict[str, int]]:
    reader = csv.DictReader(source)
    expected_header = {
        "population",
        "ordinal",
        "normalized_candidate",
        "outcome",
        "c31_emits",
        "role_llr",
        "role_signal",
        "reliability",
        "winner_margin",
        "candidate_quality",
        "country_hint",
    }
    if set(reader.fieldnames or ()) != expected_header:
        raise ValueError("unexpected proxy stream header")
    raw = list(reader)
    no_winner: dict[str, int] = {}
    winners = []
    for row in raw:
        if row["outcome"] == "no_winner":
            no_winner[row["population"]] = no_winner.get(row["population"], 0) + 1
        else:
            if row["outcome"] not in OUTCOMES:
                raise ValueError(f"unexpected proxy outcome: {row['outcome']}")
            winners.append(row)
    country_ids = {country: index for index, country in enumerate(countries)}
    logits: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(winners), BATCH_SIZE):
            batch = winners[start : start + BATCH_SIZE]
            encoded = [encode_utf8(row["normalized_candidate"]) for row in batch]
            ids = torch.tensor([value[0] for value in encoded], dtype=torch.long)
            mask = torch.tensor([value[1] for value in encoded], dtype=torch.bool)
            country = torch.tensor(
                [country_ids.get(row["country_hint"], 0) for row in batch],
                dtype=torch.long,
            )
            logits.extend(float(value) for value in model(ids, mask, country).tolist())
    scored = [
        ProxyScoredRow(
            population=row["population"],
            outcome=row["outcome"],
            logit=logit,
            score=sigmoid(logit),
            role_llr=float(row["role_llr"]),
            role_signal=float(row["role_signal"]),
            reliability=float(row["reliability"]),
            winner_margin=float(row["winner_margin"]),
            candidate_quality=float(row["candidate_quality"]),
        )
        for row, logit in zip(winners, logits)
    ]
    observed = {row.population for row in scored} | set(no_winner)
    expected = {"REAL_PROXY_V1_DEV", "REAL_PROXY_V3", "REAL_PROXY_V4"}
    if observed != expected:
        raise ValueError(f"proxy stream populations changed: {observed}")
    return scored, no_winner


def write_proxy_distributions(
    path: Path,
    rows: list[ProxyScoredRow],
    no_winner_counts: dict[str, int],
) -> None:
    populations = sorted({row.population for row in rows}) + ["COMBINED"]
    output = []
    for population in populations:
        selected_population = (
            rows
            if population == "COMBINED"
            else [row for row in rows if row.population == population]
        )
        for outcome in OUTCOMES:
            selected = [row for row in selected_population if row.outcome == outcome]
            for feature in ("score", "role_signal", "reliability", "winner_margin"):
                values = [float(getattr(row, feature)) for row in selected]
                output.append(
                    {
                        "population": population,
                        "outcome": outcome,
                        "feature": "morphology_score_uncalibrated"
                        if feature == "score"
                        else feature,
                        "rows": len(values),
                        **percentiles(values),
                    }
                )
        output.append(
            {
                "population": population,
                "outcome": "no_winner",
                "feature": "count_only",
                "rows": sum(no_winner_counts.values())
                if population == "COMBINED"
                else no_winner_counts.get(population, 0),
                **percentiles([]),
            }
        )
    write_csv(path, output)


def write_proxy_2d(path: Path, rows: list[ProxyScoredRow]) -> None:
    counts: dict[tuple[str, str, int, int], int] = {}
    for row in rows:
        score_bin = min(9, int(row.score * 10))
        role_bin = min(4, int(row.role_signal * 5))
        for population in (row.population, "COMBINED"):
            key = (population, row.outcome, score_bin, role_bin)
            counts[key] = counts.get(key, 0) + 1
    output = [
        {
            "population": population,
            "outcome": outcome,
            "neural_score_min": score_bin / 10,
            "neural_score_max": (score_bin + 1) / 10,
            "role_signal_min": role_bin / 5,
            "role_signal_max": (role_bin + 1) / 5,
            "rows": count,
        }
        for (population, outcome, score_bin, role_bin), count in sorted(counts.items())
    ]
    write_csv(path, output)


def write_proxy_correlations(path: Path, rows: list[ProxyScoredRow]) -> None:
    populations = sorted({row.population for row in rows}) + ["COMBINED"]
    output = []
    for population in populations:
        selected_population = (
            rows
            if population == "COMBINED"
            else [row for row in rows if row.population == population]
        )
        for outcome in OUTCOMES:
            selected = [row for row in selected_population if row.outcome == outcome]
            for feature in ("role_signal", "reliability", "winner_margin"):
                neural = [row.logit for row in selected]
                other = [float(getattr(row, feature)) for row in selected]
                output.append(
                    {
                        "population": population,
                        "outcome": outcome,
                        "signal": feature,
                        "rows": len(selected),
                        "pearson": pearson(neural, other),
                        "spearman": spearman(neural, other),
                    }
                )
    write_csv(path, output)


def write_proxy_thresholds(
    path: Path, rows: list[ProxyScoredRow], thresholds: dict[str, float]
) -> None:
    populations = sorted({row.population for row in rows}) + ["COMBINED"]
    output = []
    for population in populations:
        selected_population = (
            rows
            if population == "COMBINED"
            else [row for row in rows if row.population == population]
        )
        for outcome in OUTCOMES:
            for weak_only in (False, True):
                selected = [
                    row
                    for row in selected_population
                    if row.outcome == outcome
                    and (not weak_only or row.role_signal < 0.8)
                ]
                for name, threshold in thresholds.items():
                    above = sum(row.score >= threshold for row in selected)
                    output.append(
                        {
                            "population": population,
                            "outcome": outcome,
                            "role_scope": "role_signal_below_0_8"
                            if weak_only
                            else "all",
                            "threshold_name": name,
                            "threshold": threshold,
                            "rows": len(selected),
                            "above": above,
                            "rate": above / len(selected) if selected else math.nan,
                        }
                    )
    write_csv(path, output)


def proxy_summary(
    rows: list[ProxyScoredRow],
    no_winner_counts: dict[str, int],
    threshold: float,
) -> dict[str, Any]:
    populations = sorted({row.population for row in rows}) + ["COMBINED"]
    weak_rates: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for population in populations:
        selected_population = (
            rows
            if population == "COMBINED"
            else [row for row in rows if row.population == population]
        )
        weak_rates[population] = {}
        for outcome in OUTCOMES:
            selected = [
                row
                for row in selected_population
                if row.outcome == outcome and row.role_signal < 0.8
            ]
            above = sum(row.score >= threshold for row in selected)
            weak_rates[population][outcome] = {
                "rows": len(selected),
                "above": above,
                "rate": above / len(selected) if selected else None,
                "median": percentiles(row.score for row in selected)["p50"]
                if selected
                else None,
            }
    return {
        "rows_with_winner": len(rows),
        "rows_without_winner": sum(no_winner_counts.values()),
        "threshold": threshold,
        "weak_role_rates": weak_rates,
    }


def verify_selection(path: Path) -> dict[str, Any]:
    receipt = load_json(path / "selection_receipt.json")
    checks = {
        "model_sha256": path / "selected_model.f32.bin",
        "model_metadata_sha256": path / "selected_model.json",
        "thresholds_sha256": path / "frozen_thresholds.csv",
    }
    for key, file_path in checks.items():
        if receipt[key] != sha256_file(file_path):
            raise ValueError(f"selection receipt mismatch for {file_path.name}")
    return receipt


def require_split(examples: tuple[Example, ...], expected: str) -> None:
    actual = {example.split for example in examples}
    if actual != {expected}:
        raise ValueError(f"expected only {expected}, got {actual}")


def require_disjoint(*groups: tuple[Example, ...]) -> None:
    seen_exact: dict[str, int] = {}
    seen_family: dict[str, int] = {}
    for group_index, group in enumerate(groups):
        for example in group:
            prior_exact = seen_exact.setdefault(example.normalized, group_index)
            prior_family = seen_family.setdefault(example.family, group_index)
            if prior_exact != group_index:
                raise ValueError(
                    f"exact key crosses input splits: {example.normalized}"
                )
            if prior_family != group_index:
                raise ValueError(
                    f"morphology family crosses input splits: {example.family}"
                )


def read_thresholds(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    thresholds = {row["name"]: float(row["threshold"]) for row in rows}
    required = {
        "fixed_0_5",
        "zero_false_positives",
        "fpr_0_001",
        "fpr_0_005",
        "fpr_0_01",
        "precision_0_99",
        "precision_0_995",
    }
    if set(thresholds) != required:
        raise ValueError("frozen threshold set changed")
    return thresholds


def auc_row(
    scope: str,
    labels: list[int],
    scores: list[float],
    *,
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "scope": scope,
        "status": status,
        "rows": len(labels),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "roc_auc": roc_auc(labels, scores) if status == "ok" and labels else math.nan,
        "average_precision": average_precision(labels, scores)
        if status == "ok" and labels
        else math.nan,
    }


def metrics_row(
    scope: str, name: str, threshold: float, metrics: BinaryMetrics
) -> dict[str, Any]:
    return {
        "scope": scope,
        "status": "ok",
        "threshold_name": name,
        "threshold": threshold,
        **asdict(metrics),
    }


def unavailable_operating_row(
    scope: str, name: str, threshold: float
) -> dict[str, Any]:
    return {
        "scope": scope,
        "status": "unavailable",
        "threshold_name": name,
        "threshold": threshold,
        **{field: math.nan for field in BinaryMetrics.__dataclass_fields__},
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer CSV fields for empty output: {path}")
        fieldnames = rows[0].keys()
    with path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def publish_directory(output: Path, write: Callable[[Path], None]) -> None:
    if output.exists():
        raise ValueError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write(temporary)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
