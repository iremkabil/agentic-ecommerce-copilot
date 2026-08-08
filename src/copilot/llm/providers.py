"""Concrete LLM clients + a factory.

* ``OpenAICompatibleClient`` — talks to any OpenAI-compatible /chat/completions
  endpoint (hosted OpenAI, Ollama's /v1, vLLM, etc.). The wire (de)serialization
  is split into small pure helpers so they can be unit-tested without a network.
* ``ScriptedLLMClient`` — a deterministic, offline client that replays queued
  responses. It makes the Day 5 agent loop testable with zero network/model.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from copilot.llm.base import ChatMessage, LLMProviderError, LLMResponse, ToolCall, ToolSpec, Usage


def messages_to_wire(messages: list[ChatMessage]) -> list[dict]:
    """Convert our ChatMessage list into OpenAI-style message dicts."""
    wire: list[dict] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.role == "tool":
            d["tool_call_id"] = m.tool_call_id
            if m.name:
                d["name"] = m.name
        wire.append(d)
    return wire


def tools_to_wire(tools: list[ToolSpec]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def parse_response(data: dict) -> LLMResponse:
    """Parse an OpenAI-style response body into an LLMResponse."""
    message = data["choices"][0]["message"]

    tool_calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))

    usage = data.get("usage") or {}
    return LLMResponse(
        content=message.get("content"),
        tool_calls=tool_calls,
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        ),
        raw=data,
    )


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages_to_wire(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools_to_wire(tools)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # network failure, timeout, or non-2xx -- one provider-agnostic
            # error so callers never need to know this client uses httpx.
            raise LLMProviderError(f"LLM provider request failed: {exc}") from exc
        return parse_response(resp.json())


class ScriptedLLMClient:
    """Offline test double: returns queued responses in order and records calls."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if not self._responses:
            return LLMResponse(content="(no scripted response)")
        return self._responses.pop(0)


def get_llm_client(settings=None):
    """Build the LLM client selected by settings.llm_provider."""
    from copilot.config import get_settings

    s = settings or get_settings()
    provider = (s.llm_provider or "").lower()
    if provider in {"ollama", "openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleClient(
            base_url=s.llm_base_url or "http://localhost:11434/v1",
            model=s.llm_model,
            api_key=s.llm_api_key,
            temperature=s.llm_temperature,
        )
    raise ValueError(f"Unknown llm_provider: {s.llm_provider!r}")
