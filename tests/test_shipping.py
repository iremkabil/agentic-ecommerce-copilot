"""Tests for the deterministic shipping calculator."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from copilot.db.models import Base
from copilot.db.seed import load_products
from copilot.tools.shipping import compute_shipping, resolve_zone, shipping_calculator

DATA = Path(__file__).resolve().parents[1] / "data"


def test_resolve_zone():
    assert resolve_zone("US") == "domestic"
    assert resolve_zone("us") == "domestic"  # case-insensitive
    assert resolve_zone("DE") == "international"
    assert resolve_zone("JP") is None  # unsupported


def test_domestic_standard_flat_rate():
    r = compute_shipping(country="US", total_weight_g=300, subtotal=20.0, method="standard")
    assert r["cost"] == 4.90
    assert r["free_shipping_applied"] is False
    assert r["estimated_delivery_business_days"] == {"min": 3, "max": 5}


def test_domestic_free_over_threshold():
    r = compute_shipping(country="US", total_weight_g=300, subtotal=45.0, method="standard")
    assert r["cost"] == 0.0
    assert r["free_shipping_applied"] is True


def test_free_shipping_does_not_apply_to_express():
    r = compute_shipping(country="US", total_weight_g=300, subtotal=100.0, method="express")
    assert r["cost"] == 9.90
    assert r["free_shipping_applied"] is False


def test_international_weight_surcharge():
    # base 14.90 + ceil((1600-500)/500)=3 * 3.00 = 9.00 -> 23.90
    r = compute_shipping(country="DE", total_weight_g=1600, subtotal=30.0, method="standard")
    assert r["zone"] == "international"
    assert r["cost"] == 23.90
    assert r["estimated_delivery_business_days"] == {"min": 7, "max": 14}


def test_international_free_shipping_never_applies():
    r = compute_shipping(country="GB", total_weight_g=100, subtotal=500.0, method="standard")
    assert r["free_shipping_applied"] is False
    assert r["cost"] == 14.90


def test_unsupported_destination():
    r = compute_shipping(country="JP", subtotal=100.0)
    assert r["error"] == "unsupported_destination"
    assert "US" in r["supported_countries"]


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        load_products(s, DATA / "products.json")
        s.commit()
        yield s


def test_shipping_calculator_uses_db_weights_and_prices(session):
    # 2x A5 dotted notebook (260g, $14.90) + 1x bamboo organizer (780g, $29.00)
    # subtotal = 58.80 (>= 40) -> free domestic standard
    result = shipping_calculator(
        session,
        destination_country="US",
        items=[
            {"product_id": "PB-NB-001", "quantity": 2},
            {"product_id": "PB-ORG-007", "quantity": 1},
        ],
        method="standard",
    )
    assert result["subtotal"] == 58.80
    assert result["total_weight_g"] == 2 * 260 + 780
    assert result["free_shipping_applied"] is True
    assert result["cost"] == 0.0


def test_shipping_calculator_reports_unknown_products(session):
    result = shipping_calculator(
        session,
        destination_country="US",
        items=[{"product_id": "NOPE", "quantity": 1}],
        method="standard",
    )
    assert result["unknown_product_ids"] == ["NOPE"]
