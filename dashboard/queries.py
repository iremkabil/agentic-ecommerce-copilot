"""Read-only DB queries for the admin dashboard. [Day 12]

Pure functions: a Session in, plain data (dict / pandas DataFrame) out -- no
Streamlit imports here, so these are unit-testable the same way as every
other DB-touching function in this codebase (tools/*, eval/run_eval.py).
app.py is a thin rendering layer on top of this module.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from copilot.db.models import Conversation, EvalRun, GuardrailEvent, HandoffCase, Message


def kpi_summary(session: Session) -> dict:
    """Headline numbers for the KPI row: volume, handoff rate, avg latency."""
    n_conversations = session.scalar(select(func.count()).select_from(Conversation)) or 0
    n_messages = session.scalar(select(func.count()).select_from(Message)) or 0
    n_handed_off = (
        session.scalar(
            select(func.count())
            .select_from(Conversation)
            .where(Conversation.status == "handed_off")
        )
        or 0
    )
    avg_latency = session.scalar(
        select(func.avg(Message.latency_ms)).where(Message.latency_ms.is_not(None))
    )
    return {
        "conversations": n_conversations,
        "messages": n_messages,
        "handoff_rate": (n_handed_off / n_conversations) if n_conversations else 0.0,
        "avg_latency_ms": round(avg_latency) if avg_latency is not None else None,
    }


def conversations_table(session: Session, limit: int = 200) -> pd.DataFrame:
    """One row per conversation, most recent first, with its message count."""
    rows = session.execute(
        select(
            Conversation.id,
            Conversation.status,
            Conversation.created_at,
            func.count(Message.id).label("n_messages"),
        )
        .join(Message, Message.conversation_id == Conversation.id, isouter=True)
        .group_by(Conversation.id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    ).all()
    return pd.DataFrame(rows, columns=["conversation_id", "status", "created_at", "messages"])


def messages_for_conversation(session: Session, conversation_id: str) -> pd.DataFrame:
    """Full transcript (including tool calls/results) for one conversation."""
    rows = session.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
    ).all()
    return pd.DataFrame(
        [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "intent": m.intent,
                "intent_confidence": m.intent_confidence,
                "tool_name": m.tool_name,
                "guardrail_flag": m.guardrail_flag,
                "created_at": m.created_at,
            }
            for m in rows
        ]
    )


def intent_distribution(session: Session) -> pd.DataFrame:
    """Count of classified messages per intent label."""
    rows = session.execute(
        select(Message.intent, func.count().label("count"))
        .where(Message.intent.is_not(None))
        .group_by(Message.intent)
        .order_by(func.count().desc())
    ).all()
    return pd.DataFrame(rows, columns=["intent", "count"])


def tool_usage(session: Session) -> pd.DataFrame:
    """Count of tool invocations per tool name.

    Filtered to role="assistant" (the tool-call row each invocation writes,
    per agent/service.py._persist_tool_steps) so the matching role="tool"
    result row doesn't double the count.
    """
    rows = session.execute(
        select(Message.tool_name, func.count().label("count"))
        .where(Message.tool_name.is_not(None), Message.role == "assistant")
        .group_by(Message.tool_name)
        .order_by(func.count().desc())
    ).all()
    return pd.DataFrame(rows, columns=["tool_name", "count"])


def handoff_queue(session: Session, limit: int = 200) -> pd.DataFrame:
    """Handoff cases, most recent first."""
    rows = session.scalars(
        select(HandoffCase).order_by(HandoffCase.created_at.desc()).limit(limit)
    ).all()
    return pd.DataFrame(
        [
            {
                "id": h.id,
                "conversation_id": h.conversation_id,
                "reason": h.reason,
                "trigger_type": h.trigger_type,
                "priority": h.priority,
                "status": h.status,
                "created_at": h.created_at,
            }
            for h in rows
        ]
    )


def guardrail_action_breakdown(session: Session) -> pd.DataFrame:
    """Count of guardrail decisions per action (allow/block/escalate)."""
    rows = session.execute(
        select(GuardrailEvent.action, func.count().label("count")).group_by(GuardrailEvent.action)
    ).all()
    return pd.DataFrame(rows, columns=["action", "count"])


def guardrail_events_table(session: Session, limit: int = 200) -> pd.DataFrame:
    """Guardrail events, most recent first."""
    rows = session.scalars(
        select(GuardrailEvent).order_by(GuardrailEvent.created_at.desc()).limit(limit)
    ).all()
    return pd.DataFrame(
        [
            {
                "id": g.id,
                "conversation_id": g.conversation_id,
                "stage": g.stage,
                "rule": g.rule,
                "action": g.action,
                "detail": g.detail,
                "created_at": g.created_at,
            }
            for g in rows
        ]
    )


def latest_eval_run(session: Session) -> EvalRun | None:
    return session.scalars(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(1)).first()


def eval_run_history(session: Session, limit: int = 20) -> pd.DataFrame:
    """One row per eval run, oldest first (chronological, for a trend chart),
    with every metric from EvalRun.metrics flattened into its own column."""
    rows = session.scalars(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)).all()
    records = []
    for run in reversed(rows):
        record = {"run_name": run.run_name, "created_at": run.created_at}
        record.update(run.metrics or {})
        records.append(record)
    return pd.DataFrame(records)
