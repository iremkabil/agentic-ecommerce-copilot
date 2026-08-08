"""Eval runner: feed data/test_cases.csv through the live agent, compute all
6 metrics (PROJECT_PLAN.md §9), persist the run, print a summary table.
[Day 10-11]

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
* The input guardrail is likewise evaluated directly (``evaluate_input``) in
  addition to running inside ``handle_chat`` -- it's pure and free, so there's
  no reason to infer "was this blocked" from the reply text.
* Running this against the configured database is intentional, not an
  oversight: it writes real Conversation/Message/GuardrailEvent/HandoffCase
  rows the same way live traffic would, which is harmless (still 100%
  synthetic) and gives the Day 12 dashboard something to show -- including
  this run's own EvalRun/EvalResult rows.
* "passed" on an EvalResult is deliberately narrow: intent matches exactly
  AND every expected tool was called (extra tool calls don't fail a case --
  e.g. the model calling get_product_details in addition to product_search
  is still a reasonable answer). It is not a stand-in for the 6 metrics,
  which is why every metric is also stored in EvalRun.metrics.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from copilot.agent.intent import classify_intent
from copilot.agent.schemas import ChatRequest
from copilot.agent.service import get_or_build_indexes, handle_chat
from copilot.config import get_settings
from copilot.db.models import EvalResult, EvalRun, HandoffCase
from copilot.db.session import SessionLocal
from copilot.guardrails.input_rules import evaluate_input
from copilot.llm.providers import get_llm_client
from eval.metrics import (
    GuardrailMetrics,
    HandoffMetrics,
    IntentMetrics,
    MissingFieldMetrics,
    ToolSelectionMetrics,
    guardrail_metrics,
    handoff_metrics,
    intent_metrics,
    missing_field_metrics,
    order_completion_rate,
    tool_selection_metrics,
)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "test_cases.csv"

# expected_outcome values that mean "this case should actually reach the
# order-drafting logic" -- as opposed to place_order cases a guardrail blocks
# (e.g. a card number in the message) before extract/detect ever run.
_ORDER_FLOW_OUTCOMES = {"ask_missing_fields", "order_completed"}

# expected_outcome values that mean "this case should end in a handoff".
_HANDOFF_OUTCOMES = {"handoff", "escalated"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    message: str
    expected_intent: str
    expected_tools: frozenset[str]
    expected_outcome: str
    category: str
    expected_missing_fields: frozenset[str] = frozenset()
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
    expected_missing_fields: frozenset[str] | None
    predicted_missing_fields: frozenset[str] | None
    guardrail_blocked: bool
    expected_handoff: bool
    predicted_handoff: bool
    passed: bool
    reply: str


@dataclass(frozen=True)
class EvalSummary:
    n_cases: int
    intent: IntentMetrics
    tools: ToolSelectionMetrics
    order_completion: float
    missing_fields: MissingFieldMetrics
    guardrail: GuardrailMetrics
    handoff: HandoffMetrics
    results: list[CaseResult]

    def as_metrics_dict(self) -> dict:
        """Flatten every metric into one JSON-ready dict for EvalRun.metrics."""
        return {
            "n_cases": self.n_cases,
            "intent_accuracy": self.intent.accuracy,
            "intent_macro_f1": self.intent.macro_f1,
            "intent_per_label_f1": self.intent.per_label_f1,
            "tool_precision": self.tools.precision,
            "tool_recall": self.tools.recall,
            "tool_micro_f1": self.tools.micro_f1,
            "order_completion_rate": self.order_completion,
            "missing_field_precision": self.missing_fields.precision,
            "missing_field_recall": self.missing_fields.recall,
            "guardrail_block_rate": self.guardrail.block_rate,
            "guardrail_false_positive_rate": self.guardrail.false_positive_rate,
            "handoff_precision": self.handoff.precision,
            "handoff_recall": self.handoff.recall,
        }


def load_test_cases(path: Path) -> list[EvalCase]:
    """Parse test_cases.csv (PROJECT_PLAN.md §8.5, + an expected_missing_fields
    column for metric 4) into EvalCase rows."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            EvalCase(
                id=row["id"],
                message=row["message"],
                expected_intent=row["expected_intent"],
                expected_tools=frozenset(t for t in row["expected_tools"].split(";") if t),
                expected_outcome=row["expected_outcome"],
                category=row["category"],
                expected_missing_fields=frozenset(
                    t for t in (row.get("expected_missing_fields") or "").split(";") if t
                ),
                notes=row.get("notes", ""),
            )
            for row in csv.DictReader(f)
        ]


def run_case(case: EvalCase, *, session, llm, product_index, faq_index, settings) -> CaseResult:
    """Run one test case through intent classification + the full chat pipeline."""
    predicted_intent = classify_intent(
        case.message, llm=llm, confidence_threshold=settings.intent_confidence_threshold
    ).label
    guardrail_blocked = evaluate_input(case.message).action != "allow"

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

    expected_missing_fields = None
    predicted_missing_fields = None
    if case.expected_outcome in _ORDER_FLOW_OUTCOMES:
        expected_missing_fields = case.expected_missing_fields
        # last detect_missing_fields call this turn, if any; a case that never
        # got that far predicts "nothing missing", which correctly penalizes
        # recall rather than being silently excluded from the metric.
        predicted_missing_fields = frozenset()
        for step in response.steps:
            if step.tool_name == "detect_missing_fields":
                predicted_missing_fields = frozenset(step.tool_output.get("missing_fields") or [])

    expected_handoff = case.expected_outcome in _HANDOFF_OUTCOMES
    predicted_handoff = (
        session.scalars(
            select(HandoffCase).where(HandoffCase.conversation_id == response.conversation_id)
        ).first()
        is not None
    )

    passed = predicted_intent == case.expected_intent and case.expected_tools <= predicted_tools

    return CaseResult(
        id=case.id,
        message=case.message,
        category=case.category,
        expected_intent=case.expected_intent,
        predicted_intent=predicted_intent,
        expected_tools=case.expected_tools,
        predicted_tools=predicted_tools,
        order_completed=order_completed,
        expected_missing_fields=expected_missing_fields,
        predicted_missing_fields=predicted_missing_fields,
        guardrail_blocked=guardrail_blocked,
        expected_handoff=expected_handoff,
        predicted_handoff=predicted_handoff,
        passed=passed,
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
    missing_fields = missing_field_metrics(
        [
            (set(r.expected_missing_fields), set(r.predicted_missing_fields))
            for r in results
            if r.expected_missing_fields is not None
        ]
    )
    guardrail = guardrail_metrics([(r.category, r.guardrail_blocked) for r in results])
    handoff = handoff_metrics([(r.expected_handoff, r.predicted_handoff) for r in results])

    return EvalSummary(
        n_cases=len(results),
        intent=intent,
        tools=tools,
        order_completion=completion,
        missing_fields=missing_fields,
        guardrail=guardrail,
        handoff=handoff,
        results=results,
    )


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def persist_summary(session, summary: EvalSummary, *, run_name: str) -> EvalRun:
    """Write one EvalRun row + one EvalResult row per case (PROJECT_PLAN.md §7)."""
    run = EvalRun(
        run_name=run_name, git_commit=_get_git_commit(), metrics=summary.as_metrics_dict()
    )
    session.add(run)
    session.flush()  # assign run.id

    for r in summary.results:
        session.add(
            EvalResult(
                run_id=run.id,
                test_case_id=r.id,
                expected_intent=r.expected_intent,
                predicted_intent=r.predicted_intent,
                expected_tools=sorted(r.expected_tools),
                predicted_tools=sorted(r.predicted_tools),
                passed=r.passed,
                detail={
                    "category": r.category,
                    "order_completed": r.order_completed,
                    "expected_missing_fields": (
                        sorted(r.expected_missing_fields)
                        if r.expected_missing_fields is not None
                        else None
                    ),
                    "predicted_missing_fields": (
                        sorted(r.predicted_missing_fields)
                        if r.predicted_missing_fields is not None
                        else None
                    ),
                    "guardrail_blocked": r.guardrail_blocked,
                    "expected_handoff": r.expected_handoff,
                    "predicted_handoff": r.predicted_handoff,
                    "reply": r.reply,
                },
            )
        )
    session.commit()
    return run


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
    print(
        f"Missing-field:       precision {summary.missing_fields.precision:.1%}  "
        f"recall {summary.missing_fields.recall:.1%}"
    )
    print(
        f"Guardrail:           block rate {summary.guardrail.block_rate:.1%}  "
        f"false-positive rate {summary.guardrail.false_positive_rate:.1%}"
    )
    print(
        f"Handoff:             precision {summary.handoff.precision:.1%}  "
        f"recall {summary.handoff.recall:.1%}"
    )

    print("\nPer-label intent F1:")
    for label, f1 in sorted(summary.intent.per_label_f1.items()):
        print(f"  {label:<18} {f1:.3f}")

    mismatches = [r for r in summary.results if r.expected_intent != r.predicted_intent]
    if mismatches:
        print(f"\nIntent mismatches ({len(mismatches)}/{summary.n_cases}):")
        for r in mismatches:
            print(f"  {r.id}: expected={r.expected_intent!r} predicted={r.predicted_intent!r}")

    handoff_misses = [r for r in summary.results if r.expected_handoff != r.predicted_handoff]
    if handoff_misses:
        print(f"\nHandoff mismatches ({len(handoff_misses)}/{summary.n_cases}):")
        for r in handoff_misses:
            print(
                f"  {r.id}: expected_handoff={r.expected_handoff} "
                f"predicted_handoff={r.predicted_handoff}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the copilot evaluation suite.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    settings = get_settings()
    test_cases = load_test_cases(args.data_path)
    if args.limit:
        test_cases = test_cases[: args.limit]

    run_name = args.run_name or dt.datetime.now(dt.UTC).strftime("eval-%Y%m%dT%H%M%SZ")

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
        run = persist_summary(session, summary, run_name=run_name)

    print_summary(summary)
    print(f"\nPersisted as eval_runs.id={run.id} (run_name={run_name!r})")


if __name__ == "__main__":
    main()
