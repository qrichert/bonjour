from __future__ import annotations

import csv
import math
import platform
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

import torch

from .data import encode_utf8, load_json, sha256_file
from .metrics import percentiles
from .model import inference_operations, load_export, parameter_count

REQUIRED = ("Olivier", "Baris", "REDACTED", "REDACTED")
NEGATIVE_REPORT_POPULATIONS = (
    "strong_surname",
    "surname_only",
    "organization_non_name",
    "garbage_malformed",
    "dictionary_non_name",
)
HANDCRAFTED_REPORT_SIGNALS = (
    "neural_logit",
    "aggregate_role_llr",
    "aggregate_role_signal",
    "log1p_given_count",
    "log1p_total_count",
    "production_candidate_quality",
    "production_reliability",
)


def build_report(
    *,
    data: Path,
    selection: Path,
    test: Path,
    proxy: Path,
    probes: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    targets = (
        output / "qualitative_scores.csv",
        output / "deployment_estimate.csv",
        output / "latency_observation.txt",
        output / "report.md",
    )
    if any(path.exists() for path in targets):
        raise ValueError("refusing to overwrite final report outputs")
    manifest = load_json(data / "dataset_manifest.json")
    selection_receipt = load_json(selection / "selection_receipt.json")
    test_summary = load_json(test / "test_summary.json")
    proxy_summary = load_json(proxy / "proxy_summary.json")
    model, metadata = load_export(selection / "selected_model.f32.bin")
    if (
        sha256_file(selection / "selected_model.f32.bin")
        != selection_receipt["model_sha256"]
    ):
        raise ValueError("selected model changed before report generation")
    qualitative = score_probes(model, metadata, probes)
    write_qualitative(targets[0], qualitative)
    deployment = deployment_rows(model, metadata, data / "morph_test.csv", selection)
    write_rows(targets[1], deployment)
    latency = measure_latency(model)
    write_latency(targets[2], latency)
    recommendation, rationale = recommend(test_summary, proxy_summary)
    report = render_report(
        manifest,
        selection,
        test,
        selection_receipt,
        test_summary,
        proxy_summary,
        qualitative,
        deployment,
        latency,
        recommendation,
        rationale,
    )
    targets[3].write_text(report, encoding="utf-8")


def score_probes(model, metadata: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as source:
        probes = list(csv.DictReader(source))
    if {row["value"] for row in probes}.issuperset(REQUIRED) is False:
        raise ValueError("required qualitative probes are missing")
    output = []
    model.eval()
    with torch.inference_mode():
        for row in probes:
            normalized = model_normalize(row["value"])
            ids, mask, truncated = encode_utf8(normalized)
            logit = float(
                model(
                    torch.tensor([ids], dtype=torch.long),
                    torch.tensor([mask], dtype=torch.bool),
                    torch.zeros(1, dtype=torch.long),
                ).item()
            )
            output.append(
                {
                    "category": row["category"],
                    "value": row["value"],
                    "normalized": normalized,
                    "utf8_bytes": len(normalized.encode("utf-8")),
                    "truncated": truncated,
                    "country": "UNKNOWN",
                    "morphology_logit": logit,
                    "morphology_score_uncalibrated": _sigmoid(logit),
                }
            )
    return output


def deployment_rows(
    model,
    metadata: dict[str, Any],
    test_path: Path,
    selection: Path,
) -> list[dict[str, Any]]:
    lengths = []
    with test_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            lengths.append(min(96, int(row["utf8_bytes"]) + 2))
    length_percentiles = percentiles(lengths)
    parameters = parameter_count(model)
    rows = []
    for label, length in (
        ("median", int(length_percentiles["p50"])),
        ("p95", int(length_percentiles["p95"])),
        ("maximum", max(lengths)),
    ):
        operations = inference_operations(model.config, length)
        rows.append(
            {
                "model": model.config.name,
                "length_scope": label,
                "encoded_ids": length,
                "parameters": parameters,
                "float32_parameter_bytes": parameters * 4,
                "float16_parameter_bytes_estimate": parameters * 2,
                "int8_parameter_bytes_estimate": parameters,
                "float32_export_bytes": (selection / "selected_model.f32.bin")
                .stat()
                .st_size,
                **operations,
            }
        )
    return rows


def measure_latency(model) -> dict[str, Any]:
    torch.set_num_threads(1)
    ids, mask, _ = encode_utf8(model_normalize("Olivier"))
    ids_tensor = torch.tensor([ids], dtype=torch.long)
    mask_tensor = torch.tensor([mask], dtype=torch.bool)
    country = torch.zeros(1, dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        for _ in range(200):
            model(ids_tensor, mask_tensor, country)
        observations = []
        for _ in range(30):
            started = time.perf_counter_ns()
            for _ in range(100):
                model(ids_tensor, mask_tensor, country)
            observations.append((time.perf_counter_ns() - started) / 100)
    ordered = sorted(observations)
    return {
        "iterations": 3_000,
        "warmup": 200,
        "median_nanoseconds_per_token": statistics.median(ordered),
        "p95_nanoseconds_per_token": ordered[int((len(ordered) - 1) * 0.95)],
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
    }


def recommend(
    test_summary: dict[str, Any], proxy_summary: dict[str, Any]
) -> tuple[str, str]:
    combined = proxy_summary["weak_role_rates"]["COMBINED"]
    correct = combined["correct_winner_c31_abstains"]
    wrong = combined["wrong_winner"]
    null = combined["expected_null_winner"]
    correct_rate = correct["rate"] or 0.0
    wrong_rate = wrong["rate"] or 0.0
    null_rate = null["rate"] or 0.0
    gap = correct_rate - max(wrong_rate, null_rate)
    populated_fprs = [
        value
        for value in test_summary["negative_population_fpr"].values()
        if value is not None
    ]
    dangerous = any(value > 0.01 for value in populated_fprs)
    redundant = abs(test_summary["neural_role_spearman"]) >= 0.95
    reversed_or_flat = gap <= 0.0
    if dangerous or (redundant and gap < 0.10) or reversed_or_flat:
        reasons = []
        if dangerous:
            reasons.append("at least one negative population exceeds 1% FPR")
        if redundant and gap < 0.10:
            reasons.append(
                "the score is highly role-correlated without a 10-point proxy gap"
            )
        if reversed_or_flat:
            reasons.append("weak-role proxy separation is absent or reversed")
        return "discard", "; ".join(reasons)
    generations_consistent = True
    for population, outcomes in proxy_summary["weak_role_rates"].items():
        if population == "COMBINED":
            continue
        candidate = outcomes["correct_winner_c31_abstains"]
        other_rates = [
            outcomes[name]["rate"]
            for name in ("wrong_winner", "expected_null_winner")
            if outcomes[name]["rate"] is not None
        ]
        if candidate["rate"] is None or any(
            candidate["rate"] <= rate for rate in other_rates
        ):
            generations_consistent = False
    conservative = test_summary["observed_fpr_at_fpr_0_001_threshold"] <= 0.0025
    if (
        conservative
        and not dangerous
        and correct["rows"] >= 20
        and gap >= 0.10
        and generations_consistent
    ):
        return (
            "proceed to classifier-integration experiment",
            "the frozen conservative threshold separates weak-role correct abstentions by at least 10 points in every proxy generation",
        )
    return (
        "keep as experimental feature",
        "the score has some independent-looking signal, but the conservative proxy criterion is not stable or large enough for integration",
    )


def render_report(
    manifest: dict[str, Any],
    selection: Path,
    test: Path,
    selection_receipt: dict[str, Any],
    test_summary: dict[str, Any],
    proxy_summary: dict[str, Any],
    qualitative: list[dict[str, Any]],
    deployment: list[dict[str, Any]],
    latency: dict[str, Any],
    recommendation: str,
    rationale: str,
) -> str:
    configuration_rows = read_rows(selection / "config_metrics.csv")
    selected_metadata = load_json(selection / "selected_model.json")
    split_counts = manifest["split_counts"]
    required = {row["value"]: row for row in qualitative if row["value"] in REQUIRED}
    weak = proxy_summary["weak_role_rates"]["COMBINED"]
    deploy = deployment[-1]
    morphology_metrics = {
        row["scope"]: row for row in read_rows(test / "morphology_metrics.csv")
    }
    signal_metrics = {row["scope"]: row for row in read_rows(test / "signal_auc.csv")}
    population_percentiles = {
        row["population"]: row
        for row in read_rows(test / "population_percentiles.csv")
        if row["feature"] == "morphology_score_uncalibrated"
    }
    false_positive_examples = [
        row
        for row in read_rows(test / "local_case_review.csv")
        if row["slice"] == "neural_false_positive"
    ][:5]
    lines = [
        "# Tiny byte-level given-name morphology experiment",
        "",
        "This benchmark-only experiment predicts an uncalibrated morphology score from the candidate string itself. It does not change candidate generation, ranking, C3.1/C4/C5 emission, production behavior, or the compact corpus.",
        "",
        "## Training labels and split",
        "",
        "Labels were aggregated by canonicalized, Unicode-casefolded NFC string before applying `given_count >= 100 && role_llr >= +2.0` for positives and `surname_count >= 100 && role_llr <= -2.0` for strong-surname negatives. Frozen production organization vocabulary and observed lexical-gate failures supplied separate negatives. No random corruptions were used.",
        "",
        f"TRAIN contains {split_counts.get('train', 0):,} keys, VALIDATION {split_counts.get('validation', 0):,}, and MORPH_TEST {split_counts.get('morph_test', 0):,}. Exact keys and accent/case/separator families are disjoint. {manifest['stats']['quarantined_keys']:,} source keys were quarantined for qualitative diagnostics.",
        "",
        "Population membership across all splits:",
        "",
        "| Population | Rows |",
        "|---|---:|",
        *[
            f"| {population} | {rows:,} |"
            for population, rows in manifest["population_counts"].items()
        ],
        "",
        f"Surname-only negatives: **unavailable** ({manifest['surname_only_status']}). Dictionary negatives: **unavailable** ({manifest['dictionary_non_name_status']}).",
        "",
        "## Architecture selection",
        "",
        "| Configuration | Epoch | Parameters | Validation ROC AUC | Validation PR AUC | Recall at 0.1% FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in configuration_rows:
        lines.append(
            f"| {row['config']} | {row['selected_epoch']} | {int(row['parameters']):,} | {float(row['validation_roc_auc']):.4f} | {float(row['validation_average_precision']):.4f} | {float(row['validation_recall_fpr_0_001']) * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Selected `{selection_receipt['selected_config']}` at epoch {selection_receipt['selected_epoch']} with {selected_metadata['parameter_count']:,} trainable parameters. Only TRAIN/VALIDATION informed this choice.",
            "",
            "Each network embeds raw UTF-8 byte IDs, applies two ReLU 1D convolutions, performs masked global max/mean pooling, and feeds a tiny ReLU MLP. `small_country` adds a four-dimensional country embedding with a zero-vector UNKNOWN entry.",
            "",
            "## Held-out MORPH_TEST",
            "",
            f"ROC AUC: **{test_summary['roc_auc']:.4f}**. PR AUC: **{test_summary['average_precision']:.4f}**. At the validation-selected 0.1%-FPR threshold `{test_summary['fpr_0_001_threshold']:.6f}`, recall is **{test_summary['recall_at_fpr_0_001_threshold'] * 100:.2f}%** with {test_summary['false_positives_at_fpr_0_001_threshold']} false positives ({test_summary['observed_fpr_at_fpr_0_001_threshold'] * 100:.3f}% FPR).",
            "",
            f"With every country forced to UNKNOWN, ROC AUC is **{test_summary['forced_unknown_country']['roc_auc']:.4f}** and PR AUC is **{test_summary['forced_unknown_country']['average_precision']:.4f}**; recall at the same frozen threshold is **{test_summary['forced_unknown_country']['recall_at_fpr_0_001_threshold'] * 100:.2f}%**. This isolates the string contribution of the selected country-aware network.",
            "",
            f"Neural-logit versus role-signal Spearman correlation is `{test_summary['neural_role_spearman']:.4f}`. Because role LLR constructs the source labels, its standalone comparison is tautologically favorable; proxy-conditioned separation is the independence test.",
            "",
            "Population-specific morphology results at the validation-selected 0.1%-FPR threshold:",
            "",
            "| Negative population | Rows | AUC vs given | FPR | Score median | Score p95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for population in NEGATIVE_REPORT_POPULATIONS:
        metrics = morphology_metrics[f"given_vs_{population}"]
        percentile = population_percentiles[population]
        fpr = test_summary["negative_population_fpr"][population]
        lines.append(
            f"| {population} | {int(percentile['rows']):,} | "
            f"{_format_metric(metrics['roc_auc'])} | "
            f"{_format_percent(fpr)} | "
            f"{_format_metric(percentile['p50'])} | "
            f"{_format_metric(percentile['p95'])} |"
        )
    lines.extend(
        [
            "",
            "Standalone comparison with handcrafted evidence on the same frozen test:",
            "",
            "| Signal | Rows | ROC AUC | PR AUC |",
            "|---|---:|---:|---:|",
        ]
    )
    for signal in HANDCRAFTED_REPORT_SIGNALS:
        row = signal_metrics[signal]
        lines.append(
            f"| {signal} | {int(row['rows']):,} | "
            f"{_format_metric(row['roc_auc'])} | "
            f"{_format_metric(row['average_precision'])} |"
        )
    lines.extend(
        [
            "",
            "The conservative threshold still admitted strongly surname-dominant strings. Highest-scoring bounded examples were:",
            "",
            "| String | Population | Role signal | Neural score |",
            "|---|---|---:|---:|",
            *[
                f"| `{row['normalized']}` | {row['populations']} | "
                f"{float(row['role_signal']):.4f} | "
                f"{float(row['morphology_score_uncalibrated']):.6f} |"
                for row in false_positive_examples
            ],
            "",
            "Complete operating points, percentiles, handcrafted-signal correlations, conditional slices, and the bounded review are in `test/*.csv`.",
            "",
            "## Spent V1/V3/V4 proxy diagnostic",
            "",
            f"The frozen C3.1 pipeline produced {proxy_summary['rows_with_winner']:,} selected winners; {proxy_summary['rows_without_winner']:,} evaluable rows had no winner and were not scored.",
            "",
            "For weak-role winners (`role_signal < 0.8`), the validation 0.1%-FPR threshold was exceeded by:",
            "",
            "| Outcome | Rows | Above | Rate | Median score |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for outcome in (
        "correct_winner_c31_emits",
        "correct_winner_c31_abstains",
        "wrong_winner",
        "expected_null_winner",
    ):
        row = weak[outcome]
        rate = "NA" if row["rate"] is None else f"{row['rate'] * 100:.2f}%"
        median = "NA" if row["median"] is None else f"{row['median']:.4f}"
        lines.append(
            f"| {outcome} | {row['rows']} | {row['above']} | {rate} | {median} |"
        )
    lines.extend(
        [
            "",
            "The fixed neural-score/role-signal grid and correlations with reliability and winner margin are in `proxy/*.csv`. No proxy threshold or emission rule was fitted.",
            "",
            "## Required qualitative probes",
            "",
            "These strings and their morphology families were quarantined before training and selection. Scores use UNKNOWN country.",
            "",
            "| Input | Raw logit | Uncalibrated sigmoid score |",
            "|---|---:|---:|",
        ]
    )
    for value in REQUIRED:
        row = required[value]
        lines.append(
            f"| {value} | {row['morphology_logit']:.6f} | {row['morphology_score_uncalibrated']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The complete predeclared varied set is in `qualitative_scores.csv`; it did not affect the recommendation.",
            "",
            "## Deployment feasibility",
            "",
            f"The selected model has {selected_metadata['parameter_count']:,} parameters: {deploy['float32_parameter_bytes']:,} bytes of raw float32 parameters, approximately {deploy['float16_parameter_bytes_estimate']:,} bytes at float16 and {deploy['int8_parameter_bytes_estimate']:,} bytes at int8. The deterministic float32 export is {deploy['float32_export_bytes']:,} bytes including metadata.",
            "",
            f"At the MORPH_TEST maximum encoded length, inference performs about {deploy['multiply_accumulates']:,} multiply-accumulates plus {deploy['pooling_additions_comparisons']:,} pooling additions/comparisons. Observed single-thread PyTorch batch-1 latency was approximately {latency['median_nanoseconds_per_token'] / 1_000:.1f} µs/token median and {latency['p95_nanoseconds_per_token'] / 1_000:.1f} µs/token p95 on this machine. This includes framework overhead and is not a Rust production benchmark.",
            "",
            "Weights remain float32. Float16/int8 figures are estimates only; no quantization was performed.",
            "",
            "## Recommendation",
            "",
            f"**{recommendation}.** {rationale}.",
            "",
            "Stop here: this result does not implement or authorize a classifier-integration rule.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: str | float) -> str:
    numeric = float(value)
    return "NA" if math.isnan(numeric) else f"{numeric:.4f}"


def _format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100:.3f}%"


def model_normalize(value: str) -> str:
    translation = str.maketrans(
        {
            "‐": "-",
            "‑": "-",
            "‒": "-",
            "–": "-",
            "—": "-",
            "―": "-",
            "−": "-",
            "‘": "'",
            "’": "'",
            "‛": "'",
            "ʻ": "'",
            "ʼ": "'",
            "＇": "'",
        }
    )
    canonical = " ".join(
        unicodedata.normalize("NFC", value).translate(translation).split()
    )
    return unicodedata.normalize("NFC", canonical.casefold())


def write_qualitative(path: Path, rows: list[dict[str, Any]]) -> None:
    write_rows(path, rows)


def write_latency(path: Path, latency: dict[str, Any]) -> None:
    path.write_text(
        "observational only; machine-dependent\n"
        + "\n".join(f"{key}: {value}" for key, value in latency.items())
        + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
