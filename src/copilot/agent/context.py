"""Per-request execution context passed to tool executors.

Holds the DB session (per request) and the vector indexes (built once, reused).
Bundling them means tool executors have a single, typed handle to everything
they might need.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from copilot.retrieval.index import VectorIndex


@dataclass
class AgentContext:
    session: Session
    product_index: VectorIndex
    faq_index: VectorIndex
    # Set when a conversation already exists, so create_order_draft can link
    # the Order back to it. None in contexts built before a conversation id
    # is known (e.g. direct orchestrator tests).
    conversation_id: str | None = None
