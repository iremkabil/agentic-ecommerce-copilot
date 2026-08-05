"""Tests for the agent orchestrator loop, driven by the offline ScriptedLLMClient."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from copilot.agent.context import AgentContext
from copilot.agent.orchestrator import run_agent
from copilot.agent.registry import build_default_tools
from copilot.db.models import Base
from copilot.db.seed import load_products
from copilot.llm.base import LLMResponse, ToolCall
from copilot.llm.providers import ScriptedLLMClient
from copilot.retrieval.embed import HashingEmbedder
from copilot.tools.faq_retrieval import build_faq_index
from copilot.tools.product_search import build_product_index

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture()
def ctx() -> AgentContext:
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session: Session = factory()
    load_products(session, DATA / "products.json")
    session.commit()
    embedder = HashingEmbedder(dim=512)
    return AgentContext(
        session=session,
        product_index=build_product_index(session, embedder),
        faq_index=build_faq_index(embedder, DATA / "faq.md", DATA / "policies.md"),
    )


def test_agent_executes_tool_then_answers(ctx):
    llm = ScriptedLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="product_search", arguments={"query": "bamboo organizer"})]),
            LLMResponse(content="The Bamboo Desk Organizer is $29.00 and in stock."),
        ]
    )
    result = run_agent(
        "do you have a bamboo organizer?",
        llm=llm,
        tools=build_default_tools(),
        ctx=ctx,
        system_prompt="test",
        max_steps=6,
    )
    assert result.reply.startswith("The Bamboo Desk Organizer")
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "product_search"
    # the tool actually ran and found the product
    ids = [r["id"] for r in result.steps[0].tool_output["results"]]
    assert "PB-ORG-007" in ids


def test_agent_feeds_tool_result_back_to_model(ctx):
    llm = ScriptedLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="faq_retrieval", arguments={"query": "returns and refunds"})]),
            LLMResponse(content="You can return unused items within 30 days."),
        ]
    )
    run_agent("what's your return policy?", llm=llm, tools=build_default_tools(),
              ctx=ctx, system_prompt="test")
    # after the tool turn, the second llm call must have seen a role="tool" message
    second_call_roles = [m.role for m in llm.calls[1]["messages"]]
    assert "tool" in second_call_roles


def test_agent_stops_at_max_steps(ctx):
    # model that always wants another tool call -> must not loop forever
    always_tool = [
        LLMResponse(tool_calls=[ToolCall(id=str(i), name="product_search", arguments={"query": "pen"})])
        for i in range(5)
    ]
    result = run_agent("hi", llm=ScriptedLLMClient(always_tool), tools=build_default_tools(),
                       ctx=ctx, system_prompt="test", max_steps=2)
    assert len(result.steps) == 2  # exactly max_steps tool executions
    assert "human agent" in result.reply.lower()  # safe fallback


def test_agent_handles_unknown_tool(ctx):
    llm = ScriptedLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="does_not_exist", arguments={})]),
            LLMResponse(content="Sorry about that."),
        ]
    )
    result = run_agent("hi", llm=llm, tools=build_default_tools(), ctx=ctx, system_prompt="test")
    assert result.steps[0].tool_output["error"] == "unknown_tool"
    assert result.reply == "Sorry about that."
