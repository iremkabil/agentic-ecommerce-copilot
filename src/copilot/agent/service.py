"""Application service: ties the agent loop to guardrails and persistence.

Responsibilities:
* build the vector indexes once and cache them (embedding is expensive);
* load/create the conversation and its history;
* run the request lifecycle from PROJECT_PLAN.md §3: input guardrail ->
  intent -> (escalate, or run the agent) -> output guardrail -> persist;
* persist the full turn (user msg, guardrail events, each tool call + result,
  final answer) to the ``messages``/``guardrail_events``/``handoff_cases``
  tables, which is the telemetry the dashboard and eval read.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from copilot.agent.context import AgentContext
from copilot.agent.intent import classify_intent
from copilot.agent.orchestrator import AgentResult, run_agent
from copilot.agent.prompts import SYSTEM_PROMPT
from copilot.agent.registry import build_default_tools
from copilot.agent.schemas import ChatRequest, ChatResponse
from copilot.config import get_settings
from copilot.db.models import Conversation, GuardrailEvent, Message
from copilot.guardrails.escalation import escalation_from_intent
from copilot.guardrails.input_rules import evaluate_input
from copilot.guardrails.output_rules import evaluate_output
from copilot.llm.base import ChatMessage, LLMClient
from copilot.retrieval.embed import get_embedder
from copilot.retrieval.index import VectorIndex
from copilot.tools.faq_retrieval import build_faq_index
from copilot.tools.handoff import human_handoff
from copilot.tools.product_search import build_product_index

# Cache the built indexes for the process. Cleared by reset_indexes() in tests.
_INDEX_CACHE: dict[str, VectorIndex] = {}

_DEFAULT_SAFE_REPLY = (
    "I can't help with that, but I'm happy to help with products, orders, or store policy."
)
_HANDOFF_ACK = "I'm connecting you with a member of our team who can help with this."

# How much of the raw message to keep on a HandoffCase.summary -- enough for
# a human to pick up the thread without storing an unbounded blob.
_HANDOFF_SUMMARY_CHARS = 280


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


def _log_guardrail_event(
    session: Session, conversation_id: str, message_id: int, stage: str, result
) -> None:
    """Persist one guardrail decision. Called for *every* evaluation -- allow
    included -- so block rate and false-positive rate are both computable
    straight from this table (PROJECT_PLAN.md §6)."""
    session.add(
        GuardrailEvent(
            conversation_id=conversation_id,
            message_id=message_id,
            stage=stage,
            rule=result.rule,
            action=result.action,
            detail=result.detail,
        )
    )


def _persist_tool_steps(session: Session, conversation_id: str, result: AgentResult) -> None:
    for step in result.steps:
        session.add(
            Message(
                conversation_id=conversation_id,
                role="assistant",
                tool_name=step.tool_name,
                tool_input=step.tool_input,
            )
        )
        session.add(
            Message(
                conversation_id=conversation_id,
                role="tool",
                tool_name=step.tool_name,
                tool_output=step.tool_output,
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

    ctx = AgentContext(
        session=session,
        product_index=product_index,
        faq_index=faq_index,
        conversation_id=conversation.id,
    )

    user_message = Message(conversation_id=conversation.id, role="user", content=request.message)
    session.add(user_message)
    session.flush()  # assign the message id, needed by the guardrail event below

    # --- (1) input guardrail: PROJECT_PLAN.md §3 step 2, §5.4 "stop early" ---
    input_result = evaluate_input(request.message)
    _log_guardrail_event(session, conversation.id, user_message.id, "input", input_result)

    if input_result.action in ("block", "escalate"):
        if input_result.action == "escalate":
            human_handoff(
                session,
                conversation_id=conversation.id,
                reason=f"input guardrail: {input_result.rule}",
                trigger_type="guardrail",
                summary=request.message[:_HANDOFF_SUMMARY_CHARS],
                priority="high",
            )
        reply = input_result.safe_reply or _DEFAULT_SAFE_REPLY
        user_message.guardrail_flag = input_result.rule
        session.add(Message(conversation_id=conversation.id, role="assistant", content=reply))
        session.commit()
        return ChatResponse(conversation_id=conversation.id, reply=reply, steps=[])

    # --- (2) intent classification: PROJECT_PLAN.md §3 step 3 ---
    intent = classify_intent(
        request.message, llm=llm, confidence_threshold=settings.intent_confidence_threshold
    )
    user_message.intent = intent.label
    user_message.intent_confidence = intent.confidence

    trigger = escalation_from_intent(intent)
    if trigger is not None:
        human_handoff(
            session,
            conversation_id=conversation.id,
            reason=trigger.reason,
            trigger_type=trigger.trigger_type,
            summary=request.message[:_HANDOFF_SUMMARY_CHARS],
            priority=trigger.priority,
        )
        session.add(
            Message(conversation_id=conversation.id, role="assistant", content=_HANDOFF_ACK)
        )
        session.commit()
        return ChatResponse(conversation_id=conversation.id, reply=_HANDOFF_ACK, steps=[])

    # --- (3) orchestrator: PROJECT_PLAN.md §3 step 4 ---
    history = _load_history(session, conversation.id)
    result = run_agent(
        request.message,
        llm=llm,
        tools=build_default_tools(),
        ctx=ctx,
        system_prompt=SYSTEM_PROMPT,
        history=history,
        max_steps=settings.max_agent_steps,
    )

    # --- (4) output guardrail: PROJECT_PLAN.md §3 step 5 ---
    output_result = evaluate_output(result.reply, result.steps)
    final_reply = output_result.safe_reply if output_result.action == "block" else result.reply

    # --- (5) persist: PROJECT_PLAN.md §3 step 6 ---
    _persist_tool_steps(session, conversation.id, result)
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=final_reply,
        latency_ms=result.latency_ms,
        tokens_in=result.prompt_tokens,
        tokens_out=result.completion_tokens,
        guardrail_flag=output_result.rule if output_result.action == "block" else None,
    )
    session.add(assistant_message)
    session.flush()  # assign the message id, needed by the guardrail event below
    _log_guardrail_event(session, conversation.id, assistant_message.id, "output", output_result)

    session.commit()

    return ChatResponse(conversation_id=conversation.id, reply=final_reply, steps=result.steps)
