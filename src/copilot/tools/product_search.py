"""Product search tools: semantic catalog search + exact detail lookup.

Design: these are pure functions that take their dependencies (a built index,
a DB session) explicitly. That keeps them trivially testable now; the agent
(Day 5) will wire an app-level index/session once, not per call.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from copilot.db.models import Product
from copilot.retrieval.embed import Embedder
from copilot.retrieval.index import Document, VectorIndex


def _product_to_text(p: Product) -> str:
    """The text we embed for a product: name + description + category + tags.

    Concatenating the searchable fields lets a single query match on any of them.
    """
    tags = ", ".join(p.tags or [])
    return f"{p.name}. {p.description} Category: {p.category}. Tags: {tags}."


def build_product_index(session: Session, embedder: Embedder) -> VectorIndex:
    """Build a VectorIndex over every product currently in the database."""
    products = session.scalars(select(Product)).all()
    index = VectorIndex(embedder)
    index.add(
        [
            Document(
                id=p.id,
                text=_product_to_text(p),
                metadata={
                    "name": p.name,
                    "category": p.category,
                    "subcategory": p.subcategory,
                    "price": p.price,
                    "currency": p.currency,
                    "stock": p.stock,
                    "tags": p.tags or [],
                },
            )
            for p in products
        ]
    )
    return index


def search_products(
    index: VectorIndex,
    query: str,
    *,
    category: str | None = None,
    max_price: float | None = None,
    tags: list[str] | None = None,
    k: int = 5,
) -> list[dict]:
    """Semantic search with optional metadata filters. Returns JSON-ready dicts."""

    def _filter(doc: Document) -> bool:
        m = doc.metadata
        if category is not None and m.get("category") != category:
            return False
        if max_price is not None and float(m.get("price", 0.0)) > max_price:
            return False
        if tags:
            if not (set(tags) & set(m.get("tags", []))):
                return False
        return True

    hits = index.search(query, k=k, filter_fn=_filter)
    return [
        {
            "id": doc.id,
            "name": doc.metadata["name"],
            "price": doc.metadata["price"],
            "currency": doc.metadata.get("currency", "USD"),
            "category": doc.metadata["category"],
            "stock": doc.metadata.get("stock"),
            "score": round(score, 4),
        }
        for doc, score in hits
    ]


def get_product_details(session: Session, product_id: str) -> dict | None:
    """Return the full, fresh product record from the DB, or None if unknown."""
    p = session.get(Product, product_id)
    if p is None:
        return None
    return {
        "id": p.id,
        "sku": p.sku,
        "name": p.name,
        "category": p.category,
        "subcategory": p.subcategory,
        "description": p.description,
        "price": p.price,
        "currency": p.currency,
        "stock": p.stock,
        "weight_grams": p.weight_grams,
        "attributes": p.attributes,
        "tags": p.tags,
    }
