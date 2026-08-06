"""Post-agent checks: groundedness / policy / leakage / scope. [Day 9]

Same rule-based-first philosophy as input_rules.py. Groundedness is checked
against *this turn's own* tool results rather than a general fact database --
that's the one grounding signal the agent actually has available, and it's
exactly what the system prompt's GROUNDING rule already asks the model to
respect; this is the automated check on top of it.

Scope note: "output guardrail blocks the same turn twice" (PROJECT_PLAN.md
§6.3) isn't handled -- that needs a retry-capable orchestrator (re-run with
the block fed back to the model), which is out of scope for Day 9. A single
block here replaces the reply with a safe fallback rather than retrying.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from copilot.agent.prompts import SYSTEM_PROMPT
from copilot.agent.schemas import AgentStep

_UNGROUNDED_FALLBACK = "Let me double check that for you before I confirm -- one moment."

_LEAK_FALLBACK = "I can't share that, but I'm happy to help with products, orders, or store policy."

_PRICE_RE = re.compile(r"\$\s?\d+(?:\.\d{2})?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Tool/internal names that must never appear verbatim in a customer-facing reply.
_INTERNAL_NAMES = (
    "product_search",
    "get_product_details",
    "faq_retrieval",
    "shipping_calculator",
    "get_order_status",
    "extract_order_fields",
    "detect_missing_fields",
    "create_order_draft",
    "human_handoff",
    "system prompt",
)

_ADVICE_RE = re.compile(r"\b(you should (invest|sue)|guaranteed to (cure|heal|fix))\b", re.I)

_POLICY_PROMISE_RE = re.compile(
    r"\b(refund|discount|free replacement|money back|guaranteed?)\b", re.I
)

_LEAK_WINDOW = 60


@dataclass(frozen=True)
class GuardrailResult:
    """The outcome of running a drafted reply through the output rules."""

    action: str  # allow | block
    rule: str
    detail: str = ""
    safe_reply: str | None = None


def _grounded_amounts(steps: list[AgentStep]) -> set[float]:
    """All numeric values appearing anywhere in this turn's tool outputs.

    Compares by number, not string: a tool result carries a price as
    ``29.0``, not ``"$29.00"``, so matching on formatted dollar strings would
    flag every real, correctly-grounded answer as a violation. This is a
    deliberately loose heuristic (any number in the tool output "grounds"
    any dollar figure in the reply) that trades some precision for a much
    lower false-positive rate -- see PROJECT_PLAN.md §9 metric 5.
    """
    found: set[float] = set()
    for step in steps:
        for token in _NUMBER_RE.findall(str(step.tool_output)):
            found.add(round(float(token), 2))
    return found


def _reply_amounts(text: str) -> set[float]:
    amounts: set[float] = set()
    for token in _PRICE_RE.findall(text):
        amounts.add(round(float(token.replace("$", "").strip()), 2))
    return amounts


def _leaks_system_prompt(text: str) -> bool:
    if len(text) < _LEAK_WINDOW:
        return False
    return any(
        SYSTEM_PROMPT[i : i + _LEAK_WINDOW] in text
        for i in range(0, len(SYSTEM_PROMPT) - _LEAK_WINDOW, _LEAK_WINDOW // 2)
    )


def evaluate_output(reply: str, steps: list[AgentStep]) -> GuardrailResult:
    """Validate a drafted reply against this turn's tool results before it ships."""
    text = reply or ""
    lowered = text.lower()

    for name in _INTERNAL_NAMES:
        if name in lowered:
            return GuardrailResult(
                action="block", rule="prompt_leakage", detail=name, safe_reply=_LEAK_FALLBACK
            )
    if _leaks_system_prompt(text):
        return GuardrailResult(
            action="block",
            rule="prompt_leakage",
            detail="verbatim system prompt",
            safe_reply=_LEAK_FALLBACK,
        )

    if _ADVICE_RE.search(text):
        return GuardrailResult(
            action="block",
            rule="scope_violation",
            detail=_ADVICE_RE.pattern,
            safe_reply=_UNGROUNDED_FALLBACK,
        )

    if _POLICY_PROMISE_RE.search(text) and not any(s.tool_name == "faq_retrieval" for s in steps):
        return GuardrailResult(
            action="block",
            rule="ungrounded_policy_promise",
            detail=_POLICY_PROMISE_RE.pattern,
            safe_reply=_UNGROUNDED_FALLBACK,
        )

    reply_amounts = _reply_amounts(text)
    if reply_amounts and not reply_amounts & _grounded_amounts(steps):
        return GuardrailResult(
            action="block",
            rule="ungrounded_price",
            detail=", ".join(f"${a:.2f}" for a in sorted(reply_amounts)),
            safe_reply=_UNGROUNDED_FALLBACK,
        )

    return GuardrailResult(action="allow", rule="none")
