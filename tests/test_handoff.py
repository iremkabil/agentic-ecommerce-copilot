"""Tests for the human_handoff tool (Day 9): DB persistence + registry executor."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from copilot.agent.context import AgentContext
from copilot.agent.registry import build_default_tools
from copilot.db.models import Base, Conversation, HandoffCase
from copilot.tools.handoff import human_handoff


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        yield s


@pytest.fixture()
def conversation(session) -> Conversation:
    conv = Conversation()
    session.add(conv)
    session.flush()
    return conv


def test_human_handoff_persists_case_and_marks_conversation(session, conversation):
    result = human_handoff(
        session,
        conversation_id=conversation.id,
        reason="explicit human request",
        trigger_type="user_request",
        summary="Customer asked for a human.",
        priority="medium",
    )
    assert result["status"] == "open"
    assert result["trigger_type"] == "user_request"

    case = session.get(HandoffCase, result["handoff_id"])
    assert case is not None
    assert case.conversation_id == conversation.id
    assert case.status == "open"
    assert conversation.status == "handed_off"


def test_human_handoff_falls_back_on_invalid_trigger_type(session, conversation):
    result = human_handoff(
        session,
        conversation_id=conversation.id,
        reason="something",
        trigger_type="not_a_real_type",
    )
    assert result["trigger_type"] == "policy"  # safe default


def test_human_handoff_falls_back_on_invalid_priority(session, conversation):
    result = human_handoff(
        session,
        conversation_id=conversation.id,
        reason="something",
        trigger_type="policy",
        priority="urgent!!",
    )
    assert result["priority"] == "medium"  # safe default


def test_human_handoff_tolerates_unknown_conversation(session):
    # tool args come from a model; an unknown conversation_id shouldn't crash
    result = human_handoff(
        session, conversation_id="does-not-exist", reason="x", trigger_type="policy"
    )
    assert result["handoff_id"] is not None


# --- registry executor -------------------------------------------------------


@pytest.fixture()
def ctx(session, conversation) -> AgentContext:
    return AgentContext(
        session=session, product_index=None, faq_index=None, conversation_id=conversation.id
    )


def _tool(name):
    return next(t for t in build_default_tools() if t.spec.name == name)


def test_human_handoff_executor_requires_reason(ctx):
    result = _tool("human_handoff").func({}, ctx)
    assert result["error"] == "missing_required_argument"


def test_human_handoff_executor_creates_case(session, ctx, conversation):
    result = _tool("human_handoff").func(
        {"reason": "customer is upset", "trigger_type": "policy", "priority": "high"}, ctx
    )
    assert result["priority"] == "high"
    cases = session.scalars(
        select(HandoffCase).where(HandoffCase.conversation_id == conversation.id)
    ).all()
    assert len(cases) == 1
