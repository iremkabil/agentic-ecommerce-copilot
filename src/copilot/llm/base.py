"""Provider-agnostic LLM contract.

Everything above this layer (the agent, Day 5) speaks in these types and never
knows which provider is behind them. The shapes mirror the widely-used
OpenAI-style chat/tool-calling API, because Ollama and most hosted providers
expose an OpenAI-compatible endpoint — so one client covers many backends.

Flow the agent will use:
  1. send a list of ChatMessage + the available ToolSpecs
  2. get back an LLMResponse: either final ``content`` OR one/more ``tool_calls``
  3. if tool_calls: run them, append role="tool" messages, and call chat() again
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """A tool the model may call. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A model's request to invoke a tool with parsed arguments."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """One message in the conversation.

    role="tool" carries a tool result and must set ``tool_call_id`` (and usually
    ``name``) so the model can match the result to its request.
    """

    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMResponse(BaseModel):
    """What the model returned: free-text content, tool calls, or both."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] | None = None


class LLMClient(Protocol):
    """The single method the agent depends on."""

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...
