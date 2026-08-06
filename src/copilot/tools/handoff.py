"""human_handoff: create an escalation case. [Day 9]

The one DB-writing step in the handoff path, mirroring how create_order_draft
(Day 8) is the one DB-writing step in the order-drafting flow. Also marks the
conversation status=handed_off so a human queue (Day 12 dashboard) can filter
on it directly instead of re-deriving it from guardrail_events/handoff_cases
each time.

Called from two places: automatically by agent/service.py (input-guardrail
escalations, intent-driven escalations -- see guardrails/escalation.py), and
directly by the model via the human_handoff tool (agent/registry.py) for
cases the escalation matrix deliberately leaves to its judgment (serious
complaints, uncovered policy questions).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from copilot.db.models import Conversation, HandoffCase

VALID_TRIGGER_TYPES = frozenset({"user_request", "guardrail", "low_confidence", "policy"})
VALID_PRIORITIES = frozenset({"low", "medium", "high"})


def human_handoff(
    session: Session,
    *,
    conversation_id: str,
    reason: str,
    trigger_type: str,
    summary: str = "",
    priority: str = "medium",
) -> dict:
    """Create a HandoffCase and mark the conversation as handed off.

    Args come from a model in the tool-call path, not a trusted client, so
    out-of-taxonomy trigger_type/priority values fall back to a safe default
    rather than raising.
    """
    trigger_type = trigger_type if trigger_type in VALID_TRIGGER_TYPES else "policy"
    priority = priority if priority in VALID_PRIORITIES else "medium"

    case = HandoffCase(
        conversation_id=conversation_id,
        reason=reason,
        trigger_type=trigger_type,
        summary=summary,
        priority=priority,
        status="open",
    )
    session.add(case)

    conversation = session.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.status = "handed_off"

    session.flush()  # assign case.id

    return {
        "handoff_id": case.id,
        "conversation_id": conversation_id,
        "trigger_type": trigger_type,
        "priority": priority,
        "status": case.status,
    }
