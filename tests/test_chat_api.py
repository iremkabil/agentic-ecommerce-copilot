"""End-to-end test of POST /chat via FastAPI, fully offline.

Dependency overrides inject: an in-memory seeded DB, a scripted LLM, and
hashing-based indexes — so no network, model, or real endpoint is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from copilot.api.deps import get_db, get_indexes, get_llm
from copilot.api.main import app
from copilot.db.models import Base, Message
from copilot.db.seed import load_products
from copilot.llm.base import LLMResponse, ToolCall
from copilot.llm.providers import ScriptedLLMClient
from copilot.retrieval.embed import HashingEmbedder
from copilot.tools.faq_retrieval import build_faq_index
from copilot.tools.product_search import build_product_index

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},  # TestClient runs the route in a threadpool
        poolclass=StaticPool,  # one shared in-memory connection across threads
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    load_products(session, DATA / "products.json")
    session.commit()

    embedder = HashingEmbedder(dim=512)
    product_index = build_product_index(session, embedder)
    faq_index = build_faq_index(embedder, DATA / "faq.md", DATA / "policies.md")

    scripted = ScriptedLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="product_search", arguments={"query": "bamboo organizer"})]),
            LLMResponse(content="The Bamboo Desk Organizer is $29.00."),
        ]
    )

    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_llm] = lambda: scripted
    app.dependency_overrides[get_indexes] = lambda: (product_index, faq_index)

    yield TestClient(app), session

    app.dependency_overrides.clear()


def test_chat_endpoint_runs_agent_and_persists(client):
    test_client, session = client
    resp = test_client.post("/chat", json={"message": "do you have a bamboo organizer?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["reply"].startswith("The Bamboo Desk Organizer")
    assert body["conversation_id"]
    assert body["steps"][0]["tool_name"] == "product_search"

    # the turn was logged: user + assistant(tool) + tool + assistant(final) = 4 rows
    n_messages = session.scalar(select(func.count()).select_from(Message))
    assert n_messages == 4
    roles = session.scalars(select(Message.role).order_by(Message.id)).all()
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_chat_endpoint_continues_conversation(client):
    test_client, _ = client
    first = test_client.post("/chat", json={"message": "do you have a bamboo organizer?"}).json()
    conv_id = first["conversation_id"]
    # reusing the id should not error (history is reloaded); scripted llm is exhausted
    # so it returns its default final content
    second = test_client.post("/chat", json={"message": "thanks!", "conversation_id": conv_id})
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conv_id
