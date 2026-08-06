"""Pydantic schemas for the agent's public interface.

Kept separate from the LLM wire types (llm/base.py) because these are the
API-facing shapes: what a client sends to /chat and what it gets back.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class OrderItemDraft(BaseModel):
    """One line item in an in-progress order."""

    product_id: str
    quantity: int = Field(default=1, ge=1)


class OrderDraft(BaseModel):
    """Accumulated state of an order the customer is placing.

    Built up across the conversation via repeated ``extract_order_fields``
    tool calls (see tools/orders.py) and stays plain Pydantic -- no DB -- until
    ``create_order_draft`` persists it. The *model* carries this state across
    turns by re-stating what it already knows (from its own chat history)
    each time it calls the tool; the server never keeps a partial draft.
    """

    items: list[OrderItemDraft] = Field(default_factory=list)
    customer_name: str | None = None
    customer_email: str | None = None
    shipping_address_line: str | None = None
    shipping_city: str | None = None
    shipping_postal_code: str | None = None
    shipping_country: str | None = None
    shipping_method: str = "standard"


class AgentStep(BaseModel):
    """One tool invocation the agent made during a turn (for transparency/debugging)."""

    tool_name: str
    tool_input: dict
    tool_output: dict


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    steps: list[AgentStep] = Field(default_factory=list)
