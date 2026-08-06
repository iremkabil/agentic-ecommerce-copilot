"""Application service: ties the agent loop to persistence.

Responsibilities:
* build the vector indexes once and cache them (embedding is expensive);
* load/create the conversation and its history;
* run the agent;
* persist the full turn (user msg, each tool call + result, final answer) to the
  ``messages`` table, which is the telemetry the dashboard and eval read.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from copilot.agent.context import AgentContext
from copilot.agent.intent import IntentResult, classify_intent
from copilot.agent.orchestrator import AgentResult, run_agent
from copilot.agent.prompts import SYSTEM_PROMPT
from copilot.agent.registry import build_default_tools
from copilot.agent.schemas import ChatRequest, ChatResponse
from copilot.config import get_settings
from copilot.db.models import Conversation, Message
from copilot.llm.base import ChatMessage, LLMClient
from copilot.retrieval.embed import get_embedder
from copilot.retrieval.index import VectorIndex
from copilot.tools.faq_retrieval import build_faq_index
from copilot.tools.product_search import build_product_index

# Cache the built indexes for the process. Cleared by reset_indexes() in tests.
_INDEX_CACHE: dict[str, VectorIndex] = {}


def reset_indexes() -> None:
    _INDEX_CACHE.clear()


def get_or_build_indexes(session: Session, settings=None) -> tuple[VectorIndex, VectorIndex]:
    settings = settings or get_settings()
    if "product" not in _INDEX_CACHE:
        embedder = get_embedder(settings)
        _INDEX_CACHE["product"] = build_product_index(session, embedder)
        _INDEX_CACHE["faq"] = build_faq_index(
            embedder,
            settings.data_dir / "faq.md",
            settings.data_dir / "policies.md",
        )
    return _INDEX_CACHE["product"], _INDEX_CACHE["faq"]


def _load_history(session: Session, conversation_id: str, limit: int = 20) -> list[ChatMessage]:
    """Reload prior *conversational* turns (user + final assistant text).

    Tool-call and tool-result rows are excluded: they belong to a past turn's
    internal loop, not to the running dialogue the model needs as context.
    """
    rows = session.scalars(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.role.in_(["user", "assistant"]),
            Message.tool_name.is_(None),
            Message.content.is_not(None),
        )
        .order_by(Message.id)
    ).all()
    return [ChatMessage(role=r.role, content=r.content) for r in rows[-limit:]]


def _persist_turn(
    session: Session,
    conversation: Conversation,
    user_message: str,
    result: AgentResult,
    intent: IntentResult,
) -> None:
    session.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=user_message,
            intent=intent.label,
            intent_confidence=intent.confidence,
        )
    )
    for step in result.steps:
        session.add(
            Message(
                conversation_id=conversation.id,
                role="assistant",
                tool_name=step.tool_name,
                tool_input=step.tool_input,
            )
        )
        session.add(
            Message(
                conversation_id=conversation.id,
                role="tool",
                tool_name=step.tool_name,
                tool_output=step.tool_output,
            )
        )
    session.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result.reply,
            latency_ms=result.latency_ms,
            tokens_in=result.prompt_tokens,
            tokens_out=result.completion_tokens,
        )
    )


def handle_chat(
    *,
    session: Session,
    llm: LLMClient,
    request: ChatRequest,
    product_index: VectorIndex,
    faq_index: VectorIndex,
    settings=None,
) -> ChatResponse:
    settings = settings or get_settings()

    conversation = None
    if request.conversation_id:
        conversation = session.get(Conversation, request.conversation_id)
    if conversation is None:
        conversation = Conversation()
        session.add(conversation)
        session.flush()  # assign the conversation id

    history = _load_history(session, conversation.id)
    ctx = AgentContext(session=session, product_index=product_index, faq_index=faq_index)

    intent = classify_intent(
        request.message, llm=llm, confidence_threshold=settings.intent_confidence_threshold
    )

    result = run_agent(
        request.message,
        llm=llm,
        tools=build_default_tools(),
        ctx=ctx,
        system_prompt=SYSTEM_PROMPT,
        history=history,
        max_steps=settings.max_agent_steps,
    )

    _persist_turn(session, conversation, request.message, result, intent)
    session.commit()

    return ChatResponse(conversation_id=conversation.id, reply=result.reply, steps=result.steps)
