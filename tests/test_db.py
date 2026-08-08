"""Database-layer tests, run against a hermetic in-memory SQLite engine.

We don't touch the real engine/file: an in-memory database is created per test
module, so tests are fast, isolated, and leave nothing behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from copilot.db.models import (
    Base,
    Conversation,
    Customer,
    GuardrailEvent,
    HandoffCase,
    Message,
    Order,
    OrderItem,
    Product,
)
from copilot.db.seed import _seed_customers_and_orders, _seed_demo_conversations, load_products

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite://", future=True)  # in-memory
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        yield s


def test_product_roundtrip_with_json_columns(session: Session):
    session.add(
        Product(
            id="PB-TST-001",
            sku="PB-TST-001",
            name="Test Notebook",
            category="notebooks",
            price=9.99,
            stock=5,
            attributes={"pages": 100, "size": "A5"},
            tags=["test", "notebook"],
        )
    )
    session.commit()

    p = session.get(Product, "PB-TST-001")
    assert p is not None
    assert p.price == 9.99
    # JSON columns round-trip as native Python structures
    assert p.attributes["size"] == "A5"
    assert "test" in p.tags


def test_order_relationships_cascade(session: Session):
    cust = Customer(name="Test User", email="t@example.com")
    session.add(cust)
    session.add(Product(id="PB-X-1", sku="PB-X-1", name="Item", category="misc", price=5.0))
    session.flush()

    order = Order(customer_id=cust.id, status="draft", subtotal=10.0, total=10.0)
    order.items = [OrderItem(product_id="PB-X-1", quantity=2, unit_price=5.0)]
    session.add(order)
    session.commit()

    fetched = session.get(Order, order.id)
    assert fetched is not None
    assert len(fetched.items) == 1
    assert fetched.items[0].quantity == 2
    assert fetched.customer.email == "t@example.com"  # relationship navigates


def test_load_products_from_real_data_file(session: Session):
    n = load_products(session, DATA_DIR / "products.json")
    session.commit()

    count = session.scalar(select(func.count()).select_from(Product))
    assert n == count == 14  # keep in sync with data/products.json

    # spot-check a known row
    nb = session.get(Product, "PB-NB-001")
    assert nb is not None
    assert nb.category == "notebooks"
    assert nb.attributes["gsm"] == 160


def test_load_products_is_idempotent(session: Session):
    load_products(session, DATA_DIR / "products.json")
    load_products(session, DATA_DIR / "products.json")  # merge -> no duplicates
    session.commit()
    count = session.scalar(select(func.count()).select_from(Product))
    assert count == 14


def test_seed_demo_conversations_creates_a_believable_mix(session: Session):
    """The Day 14 demo-conversation seed (dashboard/screenshot data) covers a
    grounded answer, a completed order, a block, and two escalations -- one
    row of each dashboard category, so the admin dashboard has something to
    show without a live LLM."""
    load_products(session, DATA_DIR / "products.json")
    customers = _seed_customers_and_orders(session)
    _seed_demo_conversations(session, customers)
    session.commit()

    conversations = session.scalars(select(Conversation)).all()
    assert len(conversations) == 6
    assert {c.status for c in conversations} == {"active", "handed_off"}

    # every product_id referenced in the demo tool_output blobs is real
    referenced_ids = set()
    for msg in session.scalars(select(Message).where(Message.tool_output.is_not(None))).all():
        for item in msg.tool_output.get("results", msg.tool_output.get("items", [])):
            if isinstance(item, dict) and "product_id" in item:
                referenced_ids.add(item["product_id"])
            elif isinstance(item, dict) and "id" in item:
                referenced_ids.add(item["id"])
    known_ids = set(session.scalars(select(Product.id)).all())
    assert referenced_ids <= known_ids
    assert referenced_ids  # sanity: the loop above actually found some

    guardrail_actions = {g.action for g in session.scalars(select(GuardrailEvent)).all()}
    assert guardrail_actions == {"allow", "block", "escalate"}

    handoffs = session.scalars(select(HandoffCase)).all()
    assert {h.trigger_type for h in handoffs} == {"guardrail", "user_request"}
    assert all(h.status == "open" for h in handoffs)

    # blocked/escalated turns never got a classified intent ("stop early")
    blocked_user_messages = session.scalars(
        select(Message).where(Message.guardrail_flag.is_not(None))
    ).all()
    assert blocked_user_messages
    assert all(m.intent is None for m in blocked_user_messages)
