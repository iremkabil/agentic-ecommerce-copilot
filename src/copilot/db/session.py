"""Database engine and session management.

One engine per process, created from ``settings.database_url``. SQLite by
default; point the URL at Postgres and nothing else changes.

Key details
-----------
* ``check_same_thread=False`` is required only for SQLite when the connection is
  shared across threads (FastAPI's threadpool). It's a no-op for other drivers,
  so we set it conditionally.
* ``session_scope`` is a transactional context manager: it commits on success,
  rolls back on any exception, and always closes the session. This is the safe
  default pattern for writes.
* ``init_db`` uses ``create_all`` for developer convenience. In production you'd
  reach for Alembic migrations (noted as a V2 item in PROJECT_PLAN.md).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from copilot.config import get_settings
from copilot.db.models import Base

settings = get_settings()

_connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(
    settings.database_url,
    echo=False,
    future=True,
    connect_args=_connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def init_db() -> None:
    """Create all tables defined on ``Base.metadata`` if they don't exist."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope: commit on success, rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
