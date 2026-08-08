"""Offline smoke test for the Streamlit admin dashboard (Day 12).

Drives dashboard/app.py with streamlit's AppTest harness against an
in-memory SQLite DB (copilot.db.session.SessionLocal is monkeypatched) --
no real database, network, or LLM involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

from copilot.db.models import Base, Conversation, EvalRun, GuardrailEvent, HandoffCase, Message

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture(autouse=True)
def _clear_dashboard_cache():
    # _load_dashboard_data is @st.cache_data'd (by design -- see dashboard/app.py);
    # that cache is a Streamlit-process global that outlives one AppTest run, so
    # it must be cleared between tests or a later test sees an earlier test's DB.
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture()
def seeded_session_factory():
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        conv = Conversation(status="handed_off")
        session.add(conv)
        session.flush()
        session.add_all(
            [
                Message(
                    conversation_id=conv.id,
                    role="user",
                    content="I want to talk to a human",
                    intent="human_request",
                    intent_confidence=0.95,
                ),
                Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content="Connecting you now.",
                    latency_ms=120,
                ),
                GuardrailEvent(conversation_id=conv.id, stage="input", rule="none", action="allow"),
                GuardrailEvent(
                    conversation_id=conv.id,
                    stage="input",
                    rule="prompt_injection",
                    action="block",
                ),
            ]
        )
        session.add(
            HandoffCase(
                conversation_id=conv.id,
                reason="explicit human request",
                trigger_type="user_request",
                priority="medium",
                status="open",
            )
        )
        session.add(
            EvalRun(
                run_name="eval-test",
                metrics={
                    "n_cases": 71,
                    "intent_accuracy": 0.9,
                    "tool_micro_f1": 0.85,
                    "order_completion_rate": 0.8,
                    "guardrail_block_rate": 1.0,
                    "guardrail_false_positive_rate": 0.0,
                    "handoff_recall": 1.0,
                },
            )
        )
        session.commit()
    return factory


def test_dashboard_renders_without_exception_on_seeded_data(seeded_session_factory, monkeypatch):
    monkeypatch.setattr("copilot.db.session.SessionLocal", seeded_session_factory)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    metric_values = [m.value for m in at.metric]
    assert "1" in metric_values  # 1 conversation
    assert "2" in metric_values  # 2 messages

    body_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "eval-test" in body_text


def test_dashboard_renders_without_exception_on_empty_db(monkeypatch):
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    empty_factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr("copilot.db.session.SessionLocal", empty_factory)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    info_messages = " ".join(i.value for i in at.info)
    assert "No conversations yet" in info_messages
    assert "No eval runs yet" in info_messages
