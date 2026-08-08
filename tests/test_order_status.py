"""Tests for the read-only order-status tool, including its verification logic."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from copilot.db.models import Base, Customer, Order, OrderItem, Product
from copilot.tools.order_status import get_order_status


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        s.add(Product(id="PB-X-1", sku="PB-X-1", name="Item", category="misc", price=5.0))
        cust = Customer(name="Dana", email="dana@example.com")
        s.add(cust)
        s.flush()
        order = Order(
            id="ord_test_1",
            customer_id=cust.id,
            status="confirmed",
            subtotal=10.0,
            total=14.9,
            shipping_cost=4.9,
        )
        order.items = [OrderItem(product_id="PB-X-1", quantity=2, unit_price=5.0)]
        s.add(order)
        s.commit()
        yield s


def test_lookup_succeeds_with_matching_email(session):
    r = get_order_status(session, order_id="ord_test_1", email="dana@example.com")
    assert r["status"] == "confirmed"
    assert r["total"] == 14.9
    assert len(r["items"]) == 1


def test_email_is_case_insensitive(session):
    r = get_order_status(session, order_id="ord_test_1", email="  DANA@Example.com ")
    assert r["order_id"] == "ord_test_1"


def test_wrong_email_returns_generic_error(session):
    r = get_order_status(session, order_id="ord_test_1", email="attacker@example.com")
    assert r["error"] == "not_found_or_unverified"


def test_missing_order_returns_same_generic_error(session):
    # identical error to the wrong-email case -> no existence oracle
    r = get_order_status(session, order_id="ord_does_not_exist", email="dana@example.com")
    assert r["error"] == "not_found_or_unverified"
