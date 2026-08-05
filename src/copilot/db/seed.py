"""Seed the database with synthetic demo data.

Usage
-----
    python -m copilot.db.seed            # create tables, seed only if empty
    python -m copilot.db.seed --reset    # drop everything and re-seed

The product loader is factored out as ``load_products(session, path)`` so tests
can drive it against an in-memory database without touching the real engine.
Everything here is synthetic: fake customers, fake orders, fake addresses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from copilot.config import get_settings
from copilot.db.models import Base, Customer, Order, OrderItem, Product
from copilot.db.session import SessionLocal, engine, init_db

settings = get_settings()


def load_products(session: Session, path: Path) -> int:
    """Insert (or upsert) every product in ``path`` (a products.json file).

    ``session.merge`` upserts by primary key, so re-running the seed is safe and
    won't create duplicates.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in rows:
        session.merge(
            Product(
                id=row["id"],
                sku=row["sku"],
                name=row["name"],
                category=row["category"],
                subcategory=row.get("subcategory"),
                description=row.get("description", ""),
                price=row["price"],
                currency=row.get("currency", "USD"),
                stock=row.get("stock", 0),
                weight_grams=row.get("weight_grams"),
                attributes=row.get("attributes", {}),
                tags=row.get("tags", []),
            )
        )
    session.flush()
    return len(rows)


def _seed_customers_and_orders(session: Session) -> None:
    """Create a few synthetic customers and confirmed orders.

    Gives the dashboard something to render and lets `get_order_status` (Day 4)
    return real rows during development.
    """
    alice = Customer(
        name="Alice Demo", email="alice@example.com",
        address_line="12 Sample St", city="Berlin", postal_code="10115", country="DE",
    )
    bob = Customer(
        name="Bob Sample", email="bob@example.com",
        address_line="4 Placeholder Rd", city="London", postal_code="EC1A 1BB", country="GB",
    )
    carol = Customer(
        name="Carol Test", email="carol@example.com",
        address_line="88 Demo Ave", city="Austin", postal_code="73301", country="US",
    )
    session.add_all([alice, bob, carol])
    session.flush()  # assign customer ids

    # Order 1: Alice buys 2 dotted notebooks + 1 fineliner set (free shipping > $40? no -> $4.90)
    o1 = Order(
        customer_id=alice.id, status="confirmed",
        shipping_method="standard", shipping_cost=4.90, currency="USD",
    )
    o1.items = [
        OrderItem(product_id="PB-NB-001", quantity=2, unit_price=14.90),
        OrderItem(product_id="PB-PEN-014", quantity=1, unit_price=11.50),
    ]
    o1.subtotal = 2 * 14.90 + 11.50
    o1.total = round(o1.subtotal + o1.shipping_cost, 2)

    # Order 2: Bob buys a bamboo organizer + desk pad (subtotal > $40 -> free shipping)
    o2 = Order(
        customer_id=bob.id, status="confirmed",
        shipping_method="standard", shipping_cost=0.0, currency="USD",
    )
    o2.items = [
        OrderItem(product_id="PB-ORG-007", quantity=1, unit_price=29.00),
        OrderItem(product_id="PB-DESK-003", quantity=1, unit_price=34.00),
    ]
    o2.subtotal = 29.00 + 34.00
    o2.total = round(o2.subtotal + o2.shipping_cost, 2)

    session.add_all([o1, o2])
    session.flush()


def seed(reset: bool = False) -> None:
    """Create tables and populate demo data.

    If ``reset`` is True, drop all tables first. Otherwise, skip seeding when
    products already exist so accidental re-runs are harmless.
    """
    if reset:
        Base.metadata.drop_all(bind=engine)
    init_db()

    with SessionLocal() as session:
        existing = session.scalar(select(func.count()).select_from(Product))
        if existing and not reset:
            print(f"Products already present ({existing}); nothing to do. Use --reset to rebuild.")
            return

        n = load_products(session, settings.data_dir / "products.json")
        _seed_customers_and_orders(session)
        session.commit()

        customers = session.scalar(select(func.count()).select_from(Customer))
        orders = session.scalar(select(func.count()).select_from(Order))
        print(f"Seeded {n} products, {customers} customers, {orders} orders.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the copilot demo database.")
    parser.add_argument("--reset", action="store_true", help="drop all tables and re-seed")
    args = parser.parse_args()
    seed(reset=args.reset)


if __name__ == "__main__":
    main()
