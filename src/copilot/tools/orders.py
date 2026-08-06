"""Order-drafting tools: extract_order_fields / detect_missing_fields / create_order_draft.

Slot filling with no special-case state machine (PROJECT_PLAN.md §5.4): the
*model* carries the running ``OrderDraft`` across turns by re-stating whatever
it already knows -- from its own conversation history -- each time it calls
``extract_order_fields``. These functions are pure and stateless, the same
"tools are thin and deterministic" split used for ``shipping_calculator``
(``tools/shipping.py``), so no partial customer/order rows are written
mid-conversation. Only the terminal step, ``create_order_draft``, touches the
database, and only once every required field is present.

Design deviation from PROJECT_PLAN.md §5.3: the conceptual signature there is
``extract_order_fields(text, current_draft)``. Here it takes already-structured
fields instead of raw text, mirroring how ``shipping_calculator`` takes
structured ``items`` rather than a sentence -- the orchestrating LLM already
extracts structured tool arguments for every other tool, so a second, hidden
LLM call to re-parse free text would just duplicate that work.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from copilot.agent.schemas import OrderDraft
from copilot.db.models import Customer, Order, OrderItem, Product
from copilot.tools.shipping import CURRENCY

# Order-level fields required before create_order_draft will persist anything.
# ("items" is handled separately since it's a list, not a scalar.)
REQUIRED_FIELDS = (
    "items",
    "customer_name",
    "customer_email",
    "shipping_address_line",
    "shipping_country",
)

# Fields extract_order_fields will overwrite on the draft when present in a patch.
_PATCHABLE_SCALAR_FIELDS = (
    "customer_name",
    "customer_email",
    "shipping_address_line",
    "shipping_city",
    "shipping_postal_code",
    "shipping_country",
    "shipping_method",
)


def extract_order_fields(patch: dict, current: OrderDraft | None = None) -> OrderDraft:
    """Merge newly-mentioned order fields onto the running draft.

    Only fields present (and truthy) in ``patch`` overwrite ``current``;
    everything else carries over unchanged. Pure and stateless: no DB, no LLM
    call -- see the module docstring for why.
    """
    draft = (current or OrderDraft()).model_dump()
    if patch.get("items"):
        draft["items"] = patch["items"]
    for field in _PATCHABLE_SCALAR_FIELDS:
        if patch.get(field):
            draft[field] = patch[field]
    return OrderDraft.model_validate(draft)


def detect_missing_fields(draft: OrderDraft) -> list[str]:
    """Return which required fields are still empty on this draft."""
    return [field for field in REQUIRED_FIELDS if not getattr(draft, field)]


def create_order_draft(
    session: Session, draft: OrderDraft, *, conversation_id: str | None = None
) -> dict:
    """Validate and persist a complete draft as an ``Order(status="draft")``.

    Re-validates required fields and checks every item against the live
    product catalog before writing anything: the calling LLM should have
    already run ``detect_missing_fields``, but tool arguments come from a
    model, not a trusted client, so this is the last line of defense.
    """
    missing = detect_missing_fields(draft)
    if missing:
        return {"error": "missing_required_fields", "missing_fields": missing}

    order_items: list[OrderItem] = []
    subtotal = 0.0
    for item in draft.items:
        product = session.get(Product, item.product_id)
        if product is None:
            return {"error": "unknown_product_id", "product_id": item.product_id}
        if product.stock < item.quantity:
            return {
                "error": "insufficient_stock",
                "product_id": item.product_id,
                "requested": item.quantity,
                "available": product.stock,
            }
        subtotal += product.price * item.quantity
        order_items.append(
            OrderItem(product_id=product.id, quantity=item.quantity, unit_price=product.price)
        )

    email = draft.customer_email.strip().lower()
    customer = session.scalar(select(Customer).where(Customer.email == email))
    if customer is None:
        customer = Customer(name=draft.customer_name, email=email)
        session.add(customer)
    customer.name = draft.customer_name
    customer.address_line = draft.shipping_address_line
    customer.city = draft.shipping_city
    customer.postal_code = draft.shipping_postal_code
    customer.country = draft.shipping_country
    session.flush()  # assign customer.id if this is a new row

    order = Order(
        conversation_id=conversation_id,
        customer_id=customer.id,
        status="draft",
        shipping_method=draft.shipping_method,
        subtotal=round(subtotal, 2),
        total=round(subtotal, 2),  # shipping isn't costed in until shipping_calculator runs
        currency=CURRENCY,
        missing_fields=[],
        items=order_items,
    )
    session.add(order)
    session.flush()  # assign order.id

    return {
        "order_id": order.id,
        "status": order.status,
        "items": [
            {"product_id": oi.product_id, "quantity": oi.quantity, "unit_price": oi.unit_price}
            for oi in order.items
        ],
        "subtotal": order.subtotal,
        "total": order.total,
        "currency": order.currency,
    }
