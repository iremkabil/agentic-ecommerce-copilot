"""SQLAlchemy ORM models (SQLAlchemy 2.0 typed style).

Design notes
------------
* ``messages`` is the single source of truth for both the transcript *and* the
  telemetry (intent, tool call, latency, tokens). The dashboard and the eval
  harness both read from it, so we never need a parallel logging store.
* ``attributes`` / ``tags`` / ``metrics`` use JSON columns so flexible, evolving
  blobs don't require a migration for every new key. JSON works on SQLite *and*
  Postgres, which keeps the DB swap in scope.
* External-facing rows (conversations, orders) use random UUID-hex string ids so
  we don't leak sequential counts; internal rows use plain autoincrement ints.
* Status fields are plain strings with allowed values documented inline; this
  keeps SQLite painless (native ENUM types are awkward to migrate there).
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> dt.datetime:
    """Timezone-aware UTC timestamp used as the default for created_at fields."""
    return dt.datetime.now(dt.timezone.utc)


def _uuid() -> str:
    """Short, URL-safe unique id for externally-visible rows."""
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    """Declarative base; ``Base.metadata`` drives table creation."""


class Product(Base):
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # e.g. "PB-NB-001"
    sku: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    category: Mapped[str] = mapped_column(String, index=True)
    subcategory: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")
    price: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    stock: Mapped[int] = mapped_column(Integer, default=0)
    weight_grams: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    order_items: Mapped[list[OrderItem]] = relationship(back_populates="product")


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String, index=True)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)
    address_line: Mapped[str | None] = mapped_column(String, nullable=True)
    city: Mapped[str | None] = mapped_column(String, nullable=True)
    postal_code: Mapped[str | None] = mapped_column(String, nullable=True)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversations: Mapped[list[Conversation]] = relationship(back_populates="customer")
    orders: Mapped[list[Order]] = relationship(back_populates="customer")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"), nullable=True)
    channel: Mapped[str] = mapped_column(String, default="web")
    status: Mapped[str] = mapped_column(String, default="active")  # active | closed | handed_off
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    customer: Mapped[Customer | None] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    orders: Mapped[list[Order]] = relationship(back_populates="conversation")
    handoff_cases: Mapped[list[HandoffCase]] = relationship(back_populates="conversation")
    guardrail_events: Mapped[list[GuardrailEvent]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String)  # user | assistant | tool | system
    content: Mapped[str] = mapped_column(Text, default="")
    intent: Mapped[str | None] = mapped_column(String, nullable=True)
    intent_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_input: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tool_output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    guardrail_flag: Mapped[str | None] = mapped_column(String, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True
    )
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"), nullable=True)
    status: Mapped[str] = mapped_column(String, default="draft")  # draft | confirmed | cancelled
    shipping_method: Mapped[str | None] = mapped_column(String, nullable=True)
    shipping_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    subtotal: Mapped[float] = mapped_column(Float, default=0.0)
    total: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    # which required fields are still empty on a draft (drives slot filling)
    missing_fields: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    conversation: Mapped[Conversation | None] = relationship(back_populates="orders")
    customer: Mapped[Customer | None] = relationship(back_populates="orders")
    items: Mapped[list[OrderItem]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price: Mapped[float] = mapped_column(Float)

    order: Mapped[Order] = relationship(back_populates="items")
    product: Mapped[Product] = relationship(back_populates="order_items")


class HandoffCase(Base):
    __tablename__ = "handoff_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    reason: Mapped[str] = mapped_column(String)
    # user_request | guardrail | low_confidence | policy
    trigger_type: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(Text, default="")
    priority: Mapped[str] = mapped_column(String, default="medium")  # low | medium | high
    status: Mapped[str] = mapped_column(String, default="open")  # open | resolved
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="handoff_cases")


class GuardrailEvent(Base):
    __tablename__ = "guardrail_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True, index=True
    )
    message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"), nullable=True)
    stage: Mapped[str] = mapped_column(String)  # input | output
    rule: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)  # allow | block | escalate
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped[Conversation | None] = relationship(back_populates="guardrail_events")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_name: Mapped[str] = mapped_column(String)
    git_commit: Mapped[str | None] = mapped_column(String, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    results: Mapped[list[EvalResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id"), index=True)
    test_case_id: Mapped[str] = mapped_column(String)
    expected_intent: Mapped[str | None] = mapped_column(String, nullable=True)
    predicted_intent: Mapped[str | None] = mapped_column(String, nullable=True)
    expected_tools: Mapped[list] = mapped_column(JSON, default=list)
    predicted_tools: Mapped[list] = mapped_column(JSON, default=list)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    run: Mapped[EvalRun] = relationship(back_populates="results")
