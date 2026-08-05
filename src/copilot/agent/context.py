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
