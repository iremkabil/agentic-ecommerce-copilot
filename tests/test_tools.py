"""Tests for the Day 3 tools: product_search, get_product_details, faq_retrieval.

Uses a hermetic in-memory DB seeded from the real data/products.json, and the
offline HashingEmbedder. Queries are chosen to have clear lexical overlap, since
the hashing embedder is lexical rather than semantic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from copilot.db.models import Base
from copilot.db.seed import load_products
from copilot.retrieval.embed import HashingEmbedder
from copilot.tools.faq_retrieval import build_faq_index, faq_retrieval
from copilot.tools.product_search import (
    build_product_index,
    get_product_details,
    search_products,
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        load_products(s, DATA / "products.json")
        s.commit()
        yield s


@pytest.fixture()
def product_index(session):
    return build_product_index(session, HashingEmbedder(dim=512))


def test_product_search_finds_the_relevant_product(product_index):
    res = search_products(product_index, "bamboo desk organizer", k=3)
    assert res, "expected at least one hit"
    assert res[0]["id"] == "PB-ORG-007"


def test_product_search_category_filter(product_index):
    res = search_products(product_index, "pen", category="pens", k=10)
    assert res
    assert all(r["category"] == "pens" for r in res)


def test_product_search_max_price_filter(product_index):
    res = search_products(product_index, "notebook", max_price=15.0, k=10)
    assert all(r["price"] <= 15.0 for r in res)
    assert "PB-NB-003" not in [r["id"] for r in res]  # 19.50 must be excluded


def test_product_search_tags_filter(product_index):
    # PB-NB-001 is the only product tagged "bestseller", so the filter must
    # restrict results to exactly that product.
    res = search_products(product_index, "notebook", tags=["bestseller"], k=10)
    assert [r["id"] for r in res] == ["PB-NB-001"]


def test_get_product_details_returns_full_record(session):
    d = get_product_details(session, "PB-NB-001")
    assert d is not None
    assert d["name"].startswith("Paperbloom")
    assert d["attributes"]["gsm"] == 160
    assert d["tags"]  # non-empty


def test_get_product_details_unknown_id(session):
    assert get_product_details(session, "DOES-NOT-EXIST") is None


def test_faq_retrieval_finds_returns_policy():
    idx = build_faq_index(HashingEmbedder(dim=512), DATA / "faq.md", DATA / "policies.md")
    # NOTE: the hashing embedder is lexical (no stemming), so we query with the
    # surface forms that appear in the doc ("returns"/"refunds"). The real
    # sentence-transformers embedder would match "return/refund" semantically.
    res = faq_retrieval(idx, "returns and refunds", k=3)
    assert res
    titles = " ".join(r["title"].lower() for r in res)
    assert "return" in titles or "refund" in titles


def test_faq_retrieval_finds_shipping_info():
    idx = build_faq_index(HashingEmbedder(dim=512), DATA / "faq.md", DATA / "policies.md")
    res = faq_retrieval(idx, "how long does shipping take", k=3)
    assert res
    combined = " ".join(r["title"].lower() for r in res)
    assert "shipping" in combined
