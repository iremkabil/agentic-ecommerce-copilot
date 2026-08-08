"""Metric implementations: all 6 metrics from PROJECT_PLAN.md §9. [Day 10-11]

Pure functions over (expected, predicted) pairs -- no DB, no LLM, no CSV
parsing -- so they're fully offline-testable in isolation from run_eval.py,
which is the only piece that actually drives the live agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntentMetrics:
    """PROJECT_PLAN.md §9 metric 1: exact-match accuracy + macro-F1.

    Macro-F1 (mean of each label's own F1) matters alongside accuracy because
    intents are imbalanced -- a classifier that always predicts the majority
    label can still score well on accuracy alone.
    """

    accuracy: float
    macro_f1: float
    per_label_f1: dict[str, float] = field(default_factory=dict)


def intent_metrics(pairs: list[tuple[str, str]]) -> IntentMetrics:
    """``pairs`` = [(expected_label, predicted_label), ...]."""
    if not pairs:
        return IntentMetrics(accuracy=0.0, macro_f1=0.0)

    correct = sum(1 for expected, predicted in pairs if expected == predicted)
    accuracy = correct / len(pairs)

    labels = sorted({expected for expected, _ in pairs} | {predicted for _, predicted in pairs})
    per_label_f1: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for e, p in pairs if e == label and p == label)
        fp = sum(1 for e, p in pairs if e != label and p == label)
        fn = sum(1 for e, p in pairs if e == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_label_f1[label] = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )

    macro_f1 = sum(per_label_f1.values()) / len(per_label_f1)
    return IntentMetrics(accuracy=accuracy, macro_f1=macro_f1, per_label_f1=per_label_f1)


@dataclass(frozen=True)
class ToolSelectionMetrics:
    """PROJECT_PLAN.md §9 metric 2: precision/recall/micro-F1 over tool sets.

    Micro-averaged: true/false positives and false negatives are summed
    across every case *before* computing one precision/recall/F1, rather than
    averaging each case's own F1 -- this is what "micro-F1 across the suite"
    means, and it weights cases with more expected tools proportionally more.
    """

    precision: float
    recall: float
    micro_f1: float


def tool_selection_metrics(pairs: list[tuple[set[str], set[str]]]) -> ToolSelectionMetrics:
    """``pairs`` = [(expected_tools, predicted_tools), ...] as sets of tool names."""
    tp = fp = fn = 0
    for expected, predicted in pairs:
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ToolSelectionMetrics(precision=precision, recall=recall, micro_f1=micro_f1)


def order_completion_rate(completed: list[bool]) -> float:
    """PROJECT_PLAN.md §9 metric 3: share of place_order cases that reached a
    valid, persisted draft. ``completed[i]`` is True iff that case's turn
    included a successful create_order_draft call (no ``error`` key)."""
    if not completed:
        return 0.0
    return sum(completed) / len(completed)


@dataclass(frozen=True)
class MissingFieldMetrics:
    """PROJECT_PLAN.md §9 metric 4: precision/recall of detect_missing_fields
    against a case's known gaps, aggregated the same micro-averaged way as
    tool selection (metric 2)."""

    precision: float
    recall: float


def missing_field_metrics(pairs: list[tuple[set[str], set[str]]]) -> MissingFieldMetrics:
    """``pairs`` = [(expected_missing_fields, predicted_missing_fields), ...]."""
    tp = fp = fn = 0
    for expected, predicted in pairs:
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return MissingFieldMetrics(precision=precision, recall=recall)


@dataclass(frozen=True)
class GuardrailMetrics:
    """PROJECT_PLAN.md §9 metric 5: block rate on the adversarial set (recall
    of unsafe input) and false-positive rate on the benign set, reported as a
    pair -- a guardrail that blocks everything scores a perfect block rate
    but a useless false-positive rate, so neither number alone is meaningful.
    """

    block_rate: float
    false_positive_rate: float


def guardrail_metrics(flags: list[tuple[str, bool]]) -> GuardrailMetrics:
    """``flags`` = [(category, blocked_or_escalated), ...]; category is
    "benign" or "adversarial" (PROJECT_PLAN.md §8.5)."""
    adversarial = [blocked for category, blocked in flags if category == "adversarial"]
    benign = [blocked for category, blocked in flags if category == "benign"]
    block_rate = sum(adversarial) / len(adversarial) if adversarial else 0.0
    false_positive_rate = sum(benign) / len(benign) if benign else 0.0
    return GuardrailMetrics(block_rate=block_rate, false_positive_rate=false_positive_rate)


@dataclass(frozen=True)
class HandoffMetrics:
    """PROJECT_PLAN.md §9 metric 6: precision/recall of escalation decisions
    over cases labeled should-handoff vs should-not."""

    precision: float
    recall: float


def handoff_metrics(pairs: list[tuple[bool, bool]]) -> HandoffMetrics:
    """``pairs`` = [(expected_handoff, predicted_handoff), ...]."""
    tp = sum(1 for expected, predicted in pairs if expected and predicted)
    fp = sum(1 for expected, predicted in pairs if not expected and predicted)
    fn = sum(1 for expected, predicted in pairs if expected and not predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return HandoffMetrics(precision=precision, recall=recall)
