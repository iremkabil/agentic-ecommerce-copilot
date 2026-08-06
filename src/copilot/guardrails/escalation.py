"""Escalation matrix: decide when a turn should hand off to a human. [Day 9]

Pure and rule-based (PROJECT_PLAN.md §6.3), so it's fully unit-testable and
its precision/recall are measurable in the Day 10-11 eval harness.

Scope note: only two of the matrix's triggers are automatic and
pre-orchestrator here -- an explicit human request, and an intent prediction
the classifier itself isn't confident in. The rest of the matrix ("serious
complaint", "policy question not covered by retrieved docs", "output
guardrail blocks the same turn twice") is deliberately left to the *model's*
own judgment via the system prompt and the human_handoff tool it can call
directly -- the same "no special-case state machine, behavior emerges from
tools + system prompt" approach Day 8 used for order-taking (PROJECT_PLAN.md
§5.4). Auto-escalating on every complaint would also hurt the eval's
false-positive rate for a case the model is usually equipped to resolve
(e.g. by checking order status) without a human.
"""

from __future__ import annotations

from dataclasses import dataclass

from copilot.agent.intent import IntentResult


@dataclass(frozen=True)
class HandoffTrigger:
    trigger_type: str  # user_request | guardrail | low_confidence | policy
    reason: str
    priority: str  # low | medium | high


def escalation_from_intent(intent: IntentResult) -> HandoffTrigger | None:
    """Pre-orchestrator triggers driven by the Day 7 intent classifier.

    Checked in this order: an explicit request is unambiguous and always
    wins; low confidence is the fallback for "the classifier itself doesn't
    know what this is", which PROJECT_PLAN.md §3 says should bias toward
    handoff rather than letting the orchestrator guess.
    """
    if intent.label == "human_request":
        return HandoffTrigger(
            trigger_type="user_request", reason="explicit human request", priority="medium"
        )
    if intent.low_confidence:
        return HandoffTrigger(
            trigger_type="low_confidence",
            reason=f"intent confidence {intent.confidence:.2f} below threshold",
            priority="low",
        )
    return None
