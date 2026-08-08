"""Tests for the admin dashboard's DB query layer (Day 12), fully offline
against an in-memory SQLite DB -- no Streamlit involved."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from copilot.db.models import (
    Base,
    Conversation,
    EvalRun,
    GuardrailEvent,
    HandoffCase,
    Message,
)
from dashboard.queries import (
    conversations_table,
    eval_run_history,
    guardrail_action_breakdown,
    guardrail_events_table,
    handoff_queue,
    intent_distribution,
    kpi_summary,
    latest_eval_run,
    messages_for_conversation,
    tool_usage,
)


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        yield s


# --- kpi_summary --------------------------------------------------------------


def test_kpi_summary_on_empty_db(session):
    kpi = kpi_summary(session)
    assert kpi == {
        "conversations": 0,
        "messages": 0,
        "handoff_rate": 0.0,
        "avg_latency_ms": None,
    }


def test_kpi_summary_counts_and_rates(session):
    active = Conversation(status="active")
    handed_off = Conversation(status="handed_off")
    session.add_all([active, handed_off])
    session.flush()
    session.add_all(
        [
            Message(conversation_id=active.id, role="user", content="hi"),
            Message(conversation_id=active.id, role="assistant", content="hello", latency_ms=100),
            Message(conversation_id=handed_off.id, role="assistant", content="bye", latency_ms=300),
        ]
    )
    session.commit()

    kpi = kpi_summary(session)
    assert kpi["conversations"] == 2
    assert kpi["messages"] == 3
    assert kpi["handoff_rate"] == pytest.approx(0.5)
    assert kpi["avg_latency_ms"] == 200  # mean of 100 and 300; None latencies excluded


# --- conversations_table / messages_for_conversation -------------------------


def test_conversations_table_counts_messages_per_conversation(session):
    conv = Conversation(status="active")
    session.add(conv)
    session.flush()
    session.add_all(
        [
            Message(conversation_id=conv.id, role="user", content="hi"),
            Message(conversation_id=conv.id, role="assistant", content="hello"),
        ]
    )
    session.commit()

    df = conversations_table(session)
    assert len(df) == 1
    assert df.iloc[0]["conversation_id"] == conv.id
    assert df.iloc[0]["messages"] == 2


def test_conversations_table_includes_conversations_with_no_messages(session):
    conv = Conversation(status="active")
    session.add(conv)
    session.commit()

    df = conversations_table(session)
    assert len(df) == 1
    assert df.iloc[0]["messages"] == 0


def test_messages_for_conversation_returns_full_transcript_in_order(session):
    conv = Conversation(status="active")
    session.add(conv)
    session.flush()
    session.add_all(
        [
            Message(conversation_id=conv.id, role="user", content="hi", intent="greeting"),
            Message(
                conversation_id=conv.id,
                role="assistant",
                tool_name="product_search",
                tool_input={"query": "pen"},
            ),
            Message(conversation_id=conv.id, role="assistant", content="hello!"),
        ]
    )
    session.commit()

    df = messages_for_conversation(session, conv.id)
    assert list(df["role"]) == ["user", "assistant", "assistant"]
    assert df.iloc[0]["intent"] == "greeting"
    assert df.iloc[1]["tool_name"] == "product_search"


# --- intent_distribution / tool_usage -----------------------------------------


def test_intent_distribution_counts_and_excludes_nulls(session):
    conv = Conversation()
    session.add(conv)
    session.flush()
    session.add_all(
        [
            Message(conversation_id=conv.id, role="user", content="a", intent="greeting"),
            Message(conversation_id=conv.id, role="user", content="b", intent="greeting"),
            Message(conversation_id=conv.id, role="user", content="c", intent="faq_policy"),
            Message(conversation_id=conv.id, role="assistant", content="d"),  # intent=None
        ]
    )
    session.commit()

    df = intent_distribution(session)
    counts = dict(zip(df["intent"], df["count"], strict=True))
    assert counts == {"greeting": 2, "faq_policy": 1}


def test_tool_usage_counts_call_rows_not_result_rows(session):
    conv = Conversation()
    session.add(conv)
    session.flush()
    session.add_all(
        [
            Message(
                conversation_id=conv.id,
                role="assistant",
                tool_name="product_search",
                tool_input={},
            ),
            Message(
                conversation_id=conv.id,
                role="tool",
                tool_name="product_search",
                tool_output={},
            ),
        ]
    )
    session.commit()

    df = tool_usage(session)
    assert dict(zip(df["tool_name"], df["count"], strict=True)) == {"product_search": 1}


# --- handoff_queue / guardrail_* ----------------------------------------------


def test_handoff_queue_orders_most_recent_first(session):
    conv = Conversation()
    session.add(conv)
    session.flush()
    older = HandoffCase(
        conversation_id=conv.id,
        reason="r1",
        trigger_type="user_request",
        created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
    )
    newer = HandoffCase(
        conversation_id=conv.id,
        reason="r2",
        trigger_type="guardrail",
        created_at=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
    )
    session.add_all([older, newer])
    session.commit()

    df = handoff_queue(session)
    assert list(df["reason"]) == ["r2", "r1"]


def test_guardrail_action_breakdown(session):
    conv = Conversation()
    session.add(conv)
    session.flush()
    session.add_all(
        [
            GuardrailEvent(conversation_id=conv.id, stage="input", rule="none", action="allow"),
            GuardrailEvent(conversation_id=conv.id, stage="input", rule="none", action="allow"),
            GuardrailEvent(
                conversation_id=conv.id, stage="input", rule="prompt_injection", action="block"
            ),
        ]
    )
    session.commit()

    df = guardrail_action_breakdown(session)
    assert dict(zip(df["action"], df["count"], strict=True)) == {"allow": 2, "block": 1}


def test_guardrail_events_table_returns_rows(session):
    conv = Conversation()
    session.add(conv)
    session.flush()
    session.add(
        GuardrailEvent(
            conversation_id=conv.id, stage="output", rule="ungrounded_price", action="block"
        )
    )
    session.commit()

    df = guardrail_events_table(session)
    assert len(df) == 1
    assert df.iloc[0]["rule"] == "ungrounded_price"


# --- eval run queries -----------------------------------------------------------


def test_latest_eval_run_returns_none_when_empty(session):
    assert latest_eval_run(session) is None


def test_latest_eval_run_returns_most_recent(session):
    older = EvalRun(
        run_name="run-1",
        metrics={"intent_accuracy": 0.5},
        created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
    )
    newer = EvalRun(
        run_name="run-2",
        metrics={"intent_accuracy": 0.9},
        created_at=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
    )
    session.add_all([older, newer])
    session.commit()

    latest = latest_eval_run(session)
    assert latest.run_name == "run-2"


def test_eval_run_history_is_chronological_and_flattens_metrics(session):
    older = EvalRun(
        run_name="run-1",
        metrics={"intent_accuracy": 0.5},
        created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
    )
    newer = EvalRun(
        run_name="run-2",
        metrics={"intent_accuracy": 0.9},
        created_at=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
    )
    session.add_all([older, newer])
    session.commit()

    df = eval_run_history(session)
    assert list(df["run_name"]) == ["run-1", "run-2"]  # oldest first
    assert list(df["intent_accuracy"]) == [0.5, 0.9]
