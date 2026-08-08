"""Tool registry: bind the pure tool functions (Days 3-4) to LLM-facing specs.

Each Tool couples a ``ToolSpec`` (name + description + JSON-Schema parameters the
model sees) with an ``executor`` (args_dict, ctx) -> result_dict. Executors are
defensive: a model may send missing or malformed arguments, so we validate here
rather than trusting the input.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from copilot.agent.context import AgentContext
from copilot.agent.schemas import OrderDraft
from copilot.llm.base import ToolSpec
from copilot.tools.faq_retrieval import faq_retrieval
from copilot.tools.handoff import human_handoff
from copilot.tools.order_status import get_order_status
from copilot.tools.orders import create_order_draft, detect_missing_fields, extract_order_fields
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
    return (
        details if details is not None else {"error": "product_not_found", "product_id": product_id}
    )


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


def _parse_draft(value: object) -> tuple[OrderDraft | None, dict | None]:
    """Best-effort parse of a model-supplied draft object into an OrderDraft.

    Returns (draft, None) on success or (None, error_dict) on failure -- args
    come from the model, not a trusted client, so a malformed draft must
    become a tool error rather than an exception.
    """
    if not isinstance(value, dict):
        return None, {"error": "invalid_draft", "detail": "draft must be an object"}
    try:
        return OrderDraft.model_validate(value), None
    except ValidationError as exc:
        return None, {"error": "invalid_draft", "detail": str(exc)}


def _exec_extract_order_fields(args: dict, ctx: AgentContext) -> dict:
    current, error = (None, None)
    if args.get("current_draft"):
        current, error = _parse_draft(args["current_draft"])
        if error:
            return error
    draft = extract_order_fields(args, current)
    return draft.model_dump()


def _exec_detect_missing_fields(args: dict, ctx: AgentContext) -> dict:
    if not args.get("draft"):
        return {"error": "missing_required_argument", "argument": "draft"}
    draft, error = _parse_draft(args["draft"])
    if error:
        return error
    return {"missing_fields": detect_missing_fields(draft)}


def _exec_create_order_draft(args: dict, ctx: AgentContext) -> dict:
    if not args.get("draft"):
        return {"error": "missing_required_argument", "argument": "draft"}
    draft, error = _parse_draft(args["draft"])
    if error:
        return error
    return create_order_draft(ctx.session, draft, conversation_id=ctx.conversation_id)


def _exec_human_handoff(args: dict, ctx: AgentContext) -> dict:
    reason = (args.get("reason") or "").strip()
    if not reason:
        return {"error": "missing_required_argument", "argument": "reason"}
    if not ctx.conversation_id:
        return {"error": "no_active_conversation"}
    return human_handoff(
        ctx.session,
        conversation_id=ctx.conversation_id,
        reason=reason,
        trigger_type=args.get("trigger_type", "user_request"),
        summary=args.get("summary", ""),
        priority=args.get("priority", "medium"),
    )


# --- tool specs (JSON Schema the model sees) -------------------------------


def build_default_tools() -> list[Tool]:
    """Return the full V1 tool set (product/FAQ/shipping/orders + handoff)."""
    return [
        Tool(
            spec=ToolSpec(
                name="product_search",
                description="Search the Paperbloom catalog by natural-language query, "
                "with optional category, max price, and tag filters.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What the customer is looking for.",
                        },
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
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 3},
                    },
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
                        "destination_country": {
                            "type": "string",
                            "description": "ISO country code, e.g. US, DE.",
                        },
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
                        "method": {
                            "type": "string",
                            "enum": ["standard", "express"],
                            "default": "standard",
                        },
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
        Tool(
            spec=ToolSpec(
                name="extract_order_fields",
                description="Record or update the customer's in-progress order with any item, "
                "contact, or shipping details mentioned so far. Pass current_draft (the object "
                "returned by your last call to this tool) so nothing already known is lost, "
                "then add any newly-mentioned fields alongside it.",
                parameters={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "Line items known so far (include ones already "
                            "known, not just new ones).",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_id": {"type": "string"},
                                    "quantity": {"type": "integer", "default": 1},
                                },
                                "required": ["product_id"],
                            },
                        },
                        "customer_name": {"type": "string"},
                        "customer_email": {"type": "string"},
                        "shipping_address_line": {"type": "string"},
                        "shipping_city": {"type": "string"},
                        "shipping_postal_code": {"type": "string"},
                        "shipping_country": {
                            "type": "string",
                            "description": "ISO country code, e.g. US, DE, FR.",
                        },
                        "shipping_method": {
                            "type": "string",
                            "enum": ["standard", "express"],
                            "default": "standard",
                        },
                        "current_draft": {
                            "type": "object",
                            "description": "The draft object returned by your previous "
                            "extract_order_fields call, if any.",
                        },
                    },
                },
            ),
            func=_exec_extract_order_fields,
        ),
        Tool(
            spec=ToolSpec(
                name="detect_missing_fields",
                description="Given a draft from extract_order_fields, list which required "
                "fields (items, name, email, shipping address, shipping country) are still "
                "empty. Call this before create_order_draft.",
                parameters={
                    "type": "object",
                    "properties": {
                        "draft": {
                            "type": "object",
                            "description": "The draft object returned by extract_order_fields.",
                        }
                    },
                    "required": ["draft"],
                },
            ),
            func=_exec_detect_missing_fields,
        ),
        Tool(
            spec=ToolSpec(
                name="create_order_draft",
                description="Save a complete order draft. Only call this after "
                "detect_missing_fields returns an empty list for this draft.",
                parameters={
                    "type": "object",
                    "properties": {
                        "draft": {
                            "type": "object",
                            "description": "The complete draft object returned by "
                            "extract_order_fields.",
                        }
                    },
                    "required": ["draft"],
                },
            ),
            func=_exec_create_order_draft,
        ),
        Tool(
            spec=ToolSpec(
                name="human_handoff",
                description="Escalate this conversation to a human team member. Use this when "
                "the customer is abusive/threatening, has a serious complaint (e.g. damaged or "
                "wrong item), or asks a policy question faq_retrieval can't answer.",
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Short reason for the escalation.",
                        },
                        "trigger_type": {
                            "type": "string",
                            "enum": ["user_request", "guardrail", "low_confidence", "policy"],
                            "default": "user_request",
                        },
                        "summary": {
                            "type": "string",
                            "description": "One or two sentences a human agent can read to "
                            "pick up the conversation.",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "default": "medium",
                        },
                    },
                    "required": ["reason"],
                },
            ),
            func=_exec_human_handoff,
        ),
    ]
