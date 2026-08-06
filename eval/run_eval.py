"""Eval runner: feed data/test_cases.csv through the live agent, compute
metrics 1-3 (PROJECT_PLAN.md §9), print a summary table. [Day 10]

Design notes
------------
* Each row is a fresh, standalone conversation (``conversation_id=None``) --
  ``test_cases.csv`` has no multi-turn cases, so nothing needs to persist
  across rows.
* Intent is classified *twice* per row on purpose: once directly here (so
  every row gets a ``predicted_intent``, including adversarial rows the input
  guardrail blocks before ``handle_chat`` would ever classify them itself --
  see agent/service.py's "stop early" path), and once more inside
  ``handle_chat`` for non-blocked rows as part of the normal pipeline. This
  keeps intent-accuracy measurable against the *full* gold-labeled set at the
  cost of a redundant LLM call on non-blocked rows -- a fine trade at this
  scale (tens of cases, not production traffic).
* Running this against the configured database is intentional, not an
  oversight: it writes real Conversation/Message/GuardrailEvent/HandoffCase
  rows the same way live traffic would, which is harmless (still 100%
  synthetic) and gives the Day 12 dashboard something to show.
* Metrics 4-6 (missing-field detection, guardrail rates, handoff accuracy)
  and persisting to eval_runs/eval_results land Day 11.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from copilot.agent.intent import classify_intent
from copilot.agent.schemas import ChatRequest
from copilot.agent.service import get_or_build_indexes, handle_chat
from copilot.config import get_settings
from copilot.db.session import SessionLocal
from copilot.llm.providers import get_llm_client
from eval.metrics import (
    IntentMetrics,
    ToolSelectionMetrics,
    intent_metrics,
    order_completion_rate,
    tool_selection_metrics,
)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "test_cases.csv"


@dataclass(frozen=True)
class EvalCase:
    id: str
    message: str
    expected_intent: str
    expected_tools: frozenset[str]
    expected_outcome: str
    category: str
    notes: str = ""


@dataclass(frozen=True)
class CaseResult:
    id: str
    message: str
    category: str
    expected_intent: str
    predicted_intent: str
    expected_tools: frozenset[str]
    predicted_tools: frozenset[str]
    order_completed: bool | None
    reply: str


@dataclass(frozen=True)
class EvalSummary:
    n_cases: int
    intent: IntentMetrics
    tools: ToolSelectionMetrics
    order_completion: float
    results: list[CaseResult]


def load_test_cases(path: Path) -> list[EvalCase]:
    """Parse test_cases.csv (PROJECT_PLAN.md §8.5) into EvalCase rows."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            EvalCase(
                id=row["id"],
                message=row["message"],
                expected_intent=row["expected_intent"],
                expected_tools=frozenset(t for t in row["expected_tools"].split(";") if t),
                expected_outcome=row["expected_outcome"],
                category=row["category"],
                notes=row.get("notes", ""),
            )
            for row in csv.DictReader(f)
        ]


def run_case(case: EvalCase, *, session, llm, product_index, faq_index, settings) -> CaseResult:
    """Run one test case through intent classification + the full chat pipeline."""
    predicted_intent = classify_intent(
        case.message, llm=llm, confidence_threshold=settings.intent_confidence_threshold
    ).label

    response = handle_chat(
        session=session,
        llm=llm,
        request=ChatRequest(message=case.message),
        product_index=product_index,
        faq_index=faq_index,
        settings=settings,
    )
    predicted_tools = frozenset(step.tool_name for step in response.steps)

    order_completed = None
    if "create_order_draft" in case.expected_tools:
        order_completed = any(
            step.tool_name == "create_order_draft" and "error" not in step.tool_output
            for step in response.steps
        )

    return CaseResult(
        id=case.id,
        message=case.message,
        category=case.category,
        expected_intent=case.expected_intent,
        predicted_intent=predicted_intent,
        expected_tools=case.expected_tools,
        predicted_tools=predicted_tools,
        order_completed=order_completed,
        reply=response.reply,
    )


def run_eval(
    test_cases: list[EvalCase], *, session, llm, product_index, faq_index, settings=None
) -> EvalSummary:
    settings = settings or get_settings()
    results = [
        run_case(
            case,
            session=session,
            llm=llm,
            product_index=product_index,
            faq_index=faq_index,
            settings=settings,
        )
        for case in test_cases
    ]

    intent = intent_metrics([(r.expected_intent, r.predicted_intent) for r in results])
    tools = tool_selection_metrics(
        [(set(r.expected_tools), set(r.predicted_tools)) for r in results]
    )
    order_outcomes = [r.order_completed for r in results if r.order_completed is not None]
    completion = order_completion_rate(order_outcomes)

    return EvalSummary(
        n_cases=len(results),
        intent=intent,
        tools=tools,
        order_completion=completion,
        results=results,
    )


def print_summary(summary: EvalSummary) -> None:
    print(f"\n=== Eval run: {summary.n_cases} cases ===")
    print(
        f"Intent accuracy:     {summary.intent.accuracy:.1%}   "
        f"macro-F1: {summary.intent.macro_f1:.3f}"
    )
    print(
        f"Tool selection:      precision {summary.tools.precision:.1%}  "
        f"recall {summary.tools.recall:.1%}  micro-F1: {summary.tools.micro_f1:.3f}"
    )
    print(f"Order completion:    {summary.order_completion:.1%}")

    print("\nPer-label intent F1:")
    for label, f1 in sorted(summary.intent.per_label_f1.items()):
        print(f"  {label:<18} {f1:.3f}")

    mismatches = [r for r in summary.results if r.expected_intent != r.predicted_intent]
    if mismatches:
        print(f"\nIntent mismatches ({len(mismatches)}/{summary.n_cases}):")
        for r in mismatches:
            print(f"  {r.id}: expected={r.expected_intent!r} predicted={r.predicted_intent!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the copilot evaluation suite.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    args = parser.parse_args()

    settings = get_settings()
    test_cases = load_test_cases(args.data_path)
    if args.limit:
        test_cases = test_cases[: args.limit]

    with SessionLocal() as session:
        llm = get_llm_client(settings)
        product_index, faq_index = get_or_build_indexes(session, settings)
        summary = run_eval(
            test_cases,
            session=session,
            llm=llm,
            product_index=product_index,
            faq_index=faq_index,
            settings=settings,
        )

    print_summary(summary)


if __name__ == "__main__":
    main()
