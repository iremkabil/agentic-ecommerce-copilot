"""FastAPI application: health check + the agent /chat endpoint (Day 5)."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from copilot.agent.schemas import ChatRequest, ChatResponse
from copilot.agent.service import handle_chat
from copilot.api.deps import get_db, get_indexes, get_llm
from copilot.config import get_settings
from copilot.llm.base import LLMClient

settings = get_settings()

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict:
    """Liveness probe: confirms the app booted and settings loaded."""
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    session: Session = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
    indexes=Depends(get_indexes),
) -> ChatResponse:
    product_index, faq_index = indexes
    return handle_chat(
        session=session,
        llm=llm,
        request=request,
        product_index=product_index,
        faq_index=faq_index,
    )
