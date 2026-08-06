"""Tests for the Day 8 order-drafting flow: extract/detect/create + the
end-to-end orchestrator loop, all offline."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from copilot.agent.context import AgentContext
from copilot.agent.orchestrator import run_agent
from copilot.agent.registry import build_default_tools
from copilot.agent.schemas import OrderDraft
from copilot.db.models import Base, Customer, Order
from copilot.db.seed import load_products
from copilot.llm.base import LLMResponse, ToolCall
from copilot.llm.providers import ScriptedLLMClient
from copilot.retrieval.embed import HashingEmbedder
from copilot.tools.faq_retrieval import build_faq_index
from copilot.tools.orders import create_order_draft, detect_missing_fields, extract_order_fields
from copilot.tools.product_search import build_product_index

DATA = Path(__file__).resolve().parents[1] / "data"


# --- pure functions ---------------------------------------------------------


def test_extract_order_fields_merges_onto_current_draft():
    current = OrderDraft(customer_name="Alice", items=[{"product_id": "PB-NB-001", "quantity": 2}])
    updated = extract_order_fields({"customer_email": "alice@example.com"}, current)
    assert updated.customer_name == "Alice"  # carried over
    assert updated.customer_email == "alice@example.com"  # newly added
    assert updated.items[0].product_id == "PB-NB-001"  # carried over


def test_extract_order_fields_replaces_items_wholesale_when_given():
    current = OrderDraft(items=[{"product_id": "PB-NB-001", "quantity": 2}])
    updated = extract_order_fields(
        {"items": [{"product_id": "PB-ORG-007", "quantity": 1}]}, current
    )
    assert [i.product_id for i in updated.items] == ["PB-ORG-007"]


def test_extract_order_fields_ignores_falsy_patch_values():
    current = OrderDraft(customer_name="Alice")
    updated = extract_order_fields({"customer_name": ""}, current)
    assert updated.customer_name == "Alice"  # empty string doesn't clobber


def test_extract_order_fields_starts_fresh_without_current():
    updated = extract_order_fields({"customer_name": "Bob"})
    assert updated.customer_name == "Bob"
    assert updated.items == []


def test_detect_missing_fields_reports_all_when_empty():
    missing = detect_missing_fields(OrderDraft())
    assert set(missing) == {
        "items",
        "customer_name",
        "customer_email",
        "shipping_address_line",
        "shipping_country",
    }


def test_detect_missing_fields_matches_test_cases_scenario():
    # mirrors PROJECT_PLAN.md tc_002: items + country known, name/email/address missing
    draft = OrderDraft(
        items=[{"product_id": "PB-NB-001", "quantity": 3}],
        shipping_country="FR",
    )
    assert set(detect_missing_fields(draft)) == {
        "customer_name",
        "customer_email",
        "shipping_address_line",
    }


def test_detect_missing_fields_empty_when_complete():
    draft = OrderDraft(
        items=[{"product_id": "PB-NB-001", "quantity": 1}],
        customer_name="Alice",
        customer_email="alice@example.com",
        shipping_address_line="1 Rue de Paris",
        shipping_country="FR",
    )
    assert detect_missing_fields(draft) == []


# --- create_order_draft (DB-touching) ---------------------------------------


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        load_products(s, DATA / "products.json")
        s.commit()
        yield s


def _complete_draft(**overrides) -> OrderDraft:
    base = dict(
        items=[{"product_id": "PB-NB-001", "quantity": 2}],
        customer_name="Alice Example",
        customer_email="Alice@Example.com",
        shipping_address_line="1 Rue de Paris",
        shipping_country="FR",
    )
    base.update(overrides)
    return OrderDraft(**base)


def test_create_order_draft_refuses_when_fields_missing(session):
    result = create_order_draft(session, OrderDraft())
    assert result["error"] == "missing_required_fields"
    assert session.scalar(select(Order)) is None


def test_create_order_draft_persists_order_items_and_customer(session):
    result = create_order_draft(session, _complete_draft(), conversation_id="conv-1")
    assert result["status"] == "draft"
    assert result["subtotal"] == pytest.approx(29.80)  # 2 * 14.90
    assert result["total"] == result["subtotal"]
    assert result["items"] == [{"product_id": "PB-NB-001", "quantity": 2, "unit_price": 14.9}]

    order = session.get(Order, result["order_id"])
    assert order is not None
    assert order.conversation_id == "conv-1"
    assert order.customer.email == "alice@example.com"  # normalized to lowercase


def test_create_order_draft_reuses_existing_customer_by_email(session):
    create_order_draft(session, _complete_draft())
    create_order_draft(
        session, _complete_draft(items=[{"product_id": "PB-ORG-007", "quantity": 1}])
    )
    customers = session.scalars(select(Customer).where(Customer.email == "alice@example.com")).all()
    assert len(customers) == 1  # not duplicated


def test_create_order_draft_rejects_unknown_product(session):
    result = create_order_draft(
        session, _complete_draft(items=[{"product_id": "NOPE", "quantity": 1}])
    )
    assert result["error"] == "unknown_product_id"
    assert session.scalar(select(Order)) is None


def test_create_order_draft_rejects_insufficient_stock(session):
    result = create_order_draft(
        session, _complete_draft(items=[{"product_id": "PB-NB-001", "quantity": 10_000}])
    )
    assert result["error"] == "insufficient_stock"
    assert session.scalar(select(Order)) is None


# --- end-to-end via the orchestrator (mirrors PROJECT_PLAN.md tc_002) -------


@pytest.fixture()
def ctx(session) -> AgentContext:
    embedder = HashingEmbedder(dim=512)
    return AgentContext(
        session=session,
        product_index=build_product_index(session, embedder),
        faq_index=build_faq_index(embedder, DATA / "faq.md", DATA / "policies.md"),
        conversation_id="conv-order-flow",
    )


def test_agent_asks_for_missing_fields_then_completes_the_order(ctx):
    llm = ScriptedLLMClient(
        [
            # turn 1: "I want 3 dotted notebooks shipped to France"
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="extract_order_fields",
                        arguments={
                            "items": [{"product_id": "PB-NB-001", "quantity": 3}],
                            "shipping_country": "FR",
                        },
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="2",
                        name="detect_missing_fields",
                        arguments={
                            "draft": {
                                "items": [{"product_id": "PB-NB-001", "quantity": 3}],
                                "shipping_country": "FR",
                            }
                        },
                    )
                ]
            ),
            LLMResponse(content="Could I get your name, email, and shipping address?"),
        ]
    )
    result = run_agent(
        "I want 3 dotted notebooks shipped to France",
        llm=llm,
        tools=build_default_tools(),
        ctx=ctx,
        system_prompt="test",
    )
    assert "name" in result.reply.lower() or "email" in result.reply.lower()
    assert [s.tool_name for s in result.steps] == ["extract_order_fields", "detect_missing_fields"]
    assert result.steps[1].tool_output["missing_fields"] == [
        "customer_name",
        "customer_email",
        "shipping_address_line",
    ]
    # nothing was persisted yet -- the draft is still incomplete
    assert ctx.session.scalar(select(Order)) is None


def test_agent_creates_the_order_once_complete(ctx):
    complete_draft = {
        "items": [{"product_id": "PB-NB-001", "quantity": 3}],
        "shipping_country": "FR",
        "customer_name": "Alice Example",
        "customer_email": "alice@example.com",
        "shipping_address_line": "1 Rue de Paris",
    }
    llm = ScriptedLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="1", name="detect_missing_fields", arguments={"draft": complete_draft}
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(id="2", name="create_order_draft", arguments={"draft": complete_draft})
                ]
            ),
            LLMResponse(content="Order placed! Your total is $44.70."),
        ]
    )
    result = run_agent(
        "Alice Example, alice@example.com, 1 Rue de Paris",
        llm=llm,
        tools=build_default_tools(),
        ctx=ctx,
        system_prompt="test",
    )
    assert result.steps[0].tool_output["missing_fields"] == []
    create_result = result.steps[1].tool_output
    assert create_result["status"] == "draft"
    assert create_result["subtotal"] == pytest.approx(44.70)  # 3 * 14.90

    order = ctx.session.get(Order, create_result["order_id"])
    assert order is not None
    assert order.conversation_id == "conv-order-flow"
