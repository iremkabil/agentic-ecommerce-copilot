"""Tool registry: bind the pure tool functions (Days 3-4) to LLM-facing specs.

Each Tool couples a ``ToolSpec`` (name + description + JSON-Schema parameters the
model sees) with an ``executor`` (args_dict, ctx) -> result_dict. Executors are
defensive: a model may send missing or malformed arguments, so we validate here
rather than trusting the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from copilot.agent.context import AgentContext
from copilot.llm.base import ToolSpec
from copilot.tools.faq_retrieval import faq_retrieval
from copilot.tools.order_status import get_order_status
from copilot.tools.product_search import get_product_details, search_products
from copilot.tools.shipping import shipping_calculator


@dataclass
class Tool:
    spec: ToolSpec
    func: Callable[[dict, AgentContext], dict]


# --- executors -------------------------------------------------------------

def _exec_product_search(args: dict, ctx: AgentContext) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "missing_required_argument", "argument": "query"}
    return {
        "results": search_products(
            ctx.product_index,
            query,
            category=args.get("category"),
            max_price=args.get("max_price"),
            tags=args.get("tags"),
            k=int(args.get("k", 5)),
        )
    }


def _exec_get_product_details(args: dict, ctx: AgentContext) -> dict:
    product_id = (args.get("product_id") or "").strip()
    if not product_id:
        return {"error": "missing_required_argument", "argument": "product_id"}
    details = get_product_details(ctx.session, product_id)
    return details if details is not None else {"error": "product_not_found", "product_id": product_id}


def _exec_faq_retrieval(args: dict, ctx: AgentContext) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "missing_required_argument", "argument": "query"}
    return {"passages": faq_retrieval(ctx.faq_index, query, k=int(args.get("k", 3)))}


def _exec_shipping_calculator(args: dict, ctx: AgentContext) -> dict:
    country = (args.get("destination_country") or "").strip()
    items = args.get("items") or []
    if not country or not items:
        return {"error": "missing_required_argument", "argument": "destination_country/items"}
    return shipping_calculator(
        ctx.session,
        destination_country=country,
        items=items,
        method=args.get("method", "standard"),
    )


def _exec_get_order_status(args: dict, ctx: AgentContext) -> dict:
    order_id = (args.get("order_id") or "").strip()
    email = (args.get("email") or "").strip()
    if not order_id or not email:
        return {"error": "missing_required_argument", "argument": "order_id/email"}
    return get_order_status(ctx.session, order_id=order_id, email=email)


# --- tool specs (JSON Schema the model sees) -------------------------------

def build_default_tools() -> list[Tool]:
    """Return the MVP tool set. (Order-draft + handoff tools are added later.)"""
    return [
        Tool(
            spec=ToolSpec(
                name="product_search",
                description="Search the Paperbloom catalog by natural-language query, "
                "with optional category, max price, and tag filters.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What the customer is looking for."},
                        "category": {"type": "string"},
                        "max_price": {"type": "number"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            func=_exec_product_search,
        ),
        Tool(
            spec=ToolSpec(
                name="get_product_details",
                description="Get the full, current record for one product by its id.",
                parameters={
                    "type": "object",
                    "properties": {"product_id": {"type": "string"}},
                    "required": ["product_id"],
                },
            ),
            func=_exec_get_product_details,
        ),
        Tool(
            spec=ToolSpec(
                name="faq_retrieval",
                description="Retrieve relevant FAQ and store-policy passages "
                "(shipping, returns, warranty, privacy) to ground an answer.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 3}},
                    "required": ["query"],
                },
            ),
            func=_exec_faq_retrieval,
        ),
        Tool(
            spec=ToolSpec(
                name="shipping_calculator",
                description="Compute shipping cost and delivery estimate for a set of items "
                "to a destination country.",
                parameters={
                    "type": "object",
                    "properties": {
                        "destination_country": {"type": "string", "description": "ISO country code, e.g. US, DE."},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_id": {"type": "string"},
                                    "quantity": {"type": "integer"},
                                },
                                "required": ["product_id"],
                            },
                        },
                        "method": {"type": "string", "enum": ["standard", "express"], "default": "standard"},
                    },
                    "required": ["destination_country", "items"],
                },
            ),
            func=_exec_shipping_calculator,
        ),
        Tool(
            spec=ToolSpec(
                name="get_order_status",
                description="Look up an existing order. Requires both the order id and the "
                "email address on the order.",
                parameters={
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}, "email": {"type": "string"}},
                    "required": ["order_id", "email"],
                },
            ),
            func=_exec_get_order_status,
        ),
    ]
