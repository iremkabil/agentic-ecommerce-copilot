"""FastAPI dependencies.

Each external resource is its own dependency so tests can override them
individually (in-memory DB, scripted LLM, offline indexes) without touching the
route code.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from copilot.agent.service import get_or_build_indexes
from copilot.config import get_settings
from copilot.db.session import SessionLocal
from copilot.llm.base import LLMClient
from copilot.llm.providers import get_llm_client


def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_llm() -> LLMClient:
    return get_llm_client(get_settings())


def get_indexes(session: Session = Depends(get_db)):
    return get_or_build_indexes(session, get_settings())
