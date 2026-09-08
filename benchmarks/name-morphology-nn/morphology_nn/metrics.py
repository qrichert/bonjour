from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMetrics:
    rows: int
    positives: int
    negatives: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    roc_auc: float
    average_precision: float


def sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def binary_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> BinaryMetrics:
    if len(labels) != len(scores) or not labels:
        raise ValueError("binary metrics require aligned non-empty rows")
    true_positives = sum(
        label == 1 and score >= threshold for label, score in zip(labels, scores)
    )
    false_positives = sum(
        label == 0 and score >= threshold for label, score in zip(labels, scores)
    )
    positives = sum(labels)
    negatives = len(labels) - positives
    true_negatives = negatives - false_positives
    false_negatives = positives - true_positives
    emitted = true_positives + false_positives
    return BinaryMetrics(
        rows=len(labels),
        positives=positives,
        negatives=negatives,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        precision=true_positives / emitted if emitted else 1.0,
        recall=true_positives / positives if positives else math.nan,
        specificity=true_negatives / negatives if negatives else math.nan,
        false_positive_rate=false_positives / negatives if negatives else math.nan,
        roc_auc=roc_auc(labels, scores),
        average_precision=average_precision(labels, scores),
    )


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ordered = sorted(zip(scores, labels), key=lambda row: row[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return math.nan
    ordered = sorted(zip(scores, labels), key=lambda row: row[0], reverse=True)
    seen = 0
    true_positives = 0
    result = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positives = sum(label for _, label in ordered[index:end])
        seen += end - index
        true_positives += group_positives
        result += (group_positives / positives) * (true_positives / seen)
        index = end
    return result


def constrained_threshold(
    labels: list[int],
    scores: list[float],
    *,
    maximum_fpr: float | None = None,
    minimum_precision: float | None = None,
) -> tuple[float, BinaryMetrics]:
    if (maximum_fpr is None) == (minimum_precision is None):
        raise ValueError("select exactly one threshold constraint")
    if len(labels) != len(scores) or not labels:
        raise ValueError("threshold selection requires aligned non-empty rows")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("threshold selection requires both labels")
    ordered = sorted(zip(scores, labels), key=lambda row: row[0], reverse=True)
    selected_threshold = math.nextafter(ordered[0][0], math.inf)
    selected_true_positives = 0
    selected_false_positives = 0
    true_positives = 0
    false_positives = 0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positives = sum(label for _, label in ordered[index:end])
        true_positives += group_positives
        false_positives += end - index - group_positives
        emitted = true_positives + false_positives
        false_positive_rate = false_positives / negatives
        precision = true_positives / emitted
        feasible = (maximum_fpr is not None and false_positive_rate <= maximum_fpr) or (
            minimum_precision is not None and precision >= minimum_precision
        )
        if feasible and true_positives >= selected_true_positives:
            selected_threshold = ordered[index][0]
            selected_true_positives = true_positives
            selected_false_positives = false_positives
        index = end
    metrics = binary_metrics(labels, scores, selected_threshold)
    if (
        metrics.true_positives != selected_true_positives
        or metrics.false_positives != selected_false_positives
    ):
        raise AssertionError("threshold sweep counts disagree with exact metrics")
    return selected_threshold, metrics


def percentiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {f"p{point}": math.nan for point in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    return {
        f"p{point}": ordered[((len(ordered) - 1) * point) // 100]
        for point in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = sum((value - left_mean) ** 2 for value in left)
    right_scale = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else math.nan


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda row: row[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        for original, _ in ordered[index:end]:
            ranks[original] = rank
        index = end
    return ranks
