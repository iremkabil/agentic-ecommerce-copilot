"""Pydantic schemas for the agent's public interface.

Kept separate from the LLM wire types (llm/base.py) because these are the
API-facing shapes: what a client sends to /chat and what it gets back.

(OrderDraft lands here on Day 8 with the order-taking flow.)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class AgentStep(BaseModel):
    """One tool invocation the agent made during a turn (for transparency/debugging)."""

    tool_name: str
    tool_input: dict
    tool_output: dict


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    steps: list[AgentStep] = Field(default_factory=list)
