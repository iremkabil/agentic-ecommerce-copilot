"""Order-status lookup tool (read-only).

Requires both an order id *and* the matching email. This is a small but real
security detail: without it, anyone who guessed an order id could read someone
else's order. On any mismatch we return the *same* generic error whether the
order is missing or the email is wrong, so the tool can't be used as an oracle
to enumerate which order ids exist.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from copilot.db.models import Order

_GENERIC_ERROR = {
    "error": "not_found_or_unverified",
    "message": "No order matches that id and email.",
}


def get_order_status(session: Session, *, order_id: str, email: str) -> dict:
    """Return the status of an order if the id + email match, else a generic error."""
    order = session.get(Order, order_id)
    email_norm = (email or "").strip().lower()

    if order is None or order.customer is None or order.customer.email.lower() != email_norm:
        return dict(_GENERIC_ERROR)

    return {
        "order_id": order.id,
        "status": order.status,
        "shipping_method": order.shipping_method,
        "shipping_cost": order.shipping_cost,
        "subtotal": order.subtotal,
        "total": order.total,
        "currency": order.currency,
        "items": [
            {
                "product_id": item.product_id,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
            }
            for item in order.items
        ],
        "created_at": order.created_at.isoformat() if order.created_at else None,
    }
