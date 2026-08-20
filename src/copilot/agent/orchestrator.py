"""The agent loop — native tool calling.

This is the mechanic behind every modern agent, written out plainly so you can
explain it in an interview:

    build messages (system + history + user)
    repeat up to max_steps:
        response = llm.chat(messages, tools)
        if response has tool_calls:
            append the assistant tool-call message
            run each tool, append a role="tool" result message
            continue the loop  (the model now sees the results)
        else:
            response.content is the final answer -> stop
    if we ran out of steps without a final answer -> safe fallback

No framework is hiding this. Day 5 keeps it a hand-written loop on purpose; V1
can lift it into LangGraph, but the semantics stay identical.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from copilot.agent.context import AgentContext
from copilot.agent.registry import Tool
from copilot.agent.schemas import AgentStep
from copilot.llm.base import ChatMessage, LLMClient, LLMProviderError, ToolSpec

_FALLBACK_REPLY = (
    "I'm having trouble completing that right now. Let me connect you with a human agent."
)


@dataclass
class AgentResult:
    reply: str
    steps: list[AgentStep] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


def _run_tool(tool: Tool | None, name: str, args: dict, ctx: AgentContext) -> dict:
    """Execute one tool, converting any failure into a structured result."""
    if tool is None:
        return {"error": "unknown_tool", "tool": name}
    try:
        return tool.func(args, ctx)
    except Exception as exc:  # never let a tool crash the whole turn
        return {"error": "tool_execution_failed", "detail": str(exc)}


def run_agent(
    user_message: str,
    *,
    llm: LLMClient,
    tools: list[Tool],
    ctx: AgentContext,
    system_prompt: str,
    history: list[ChatMessage] | None = None,
    max_steps: int = 6,
) -> AgentResult:
    tools_by_name = {t.spec.name: t for t in tools}
    tool_specs: list[ToolSpec] = [t.spec for t in tools]

    messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
    messages.extend(history or [])
    messages.append(ChatMessage(role="user", content=user_message))

    steps: list[AgentStep] = []
    prompt_tokens = completion_tokens = 0
    reply = ""
    start = time.perf_counter()

    for _ in range(max_steps):
        try:
            response = llm.chat(messages, tools=tool_specs)
        except LLMProviderError as exc:
            print("LLMProviderError:", exc)
            reply = _FALLBACK_REPLY
            break
        prompt_tokens += response.usage.prompt_tokens
        completion_tokens += response.usage.completion_tokens

        if response.tool_calls:
            # record the model's tool-call turn, then execute each call
            messages.append(
                ChatMessage(
                    role="assistant", content=response.content, tool_calls=response.tool_calls
                )
            )
            for call in response.tool_calls:
                result = _run_tool(tools_by_name.get(call.name), call.name, call.arguments, ctx)
                steps.append(
                    AgentStep(tool_name=call.name, tool_input=call.arguments, tool_output=result)
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call.id,
                        name=call.name,
                        content=json.dumps(result),
                    )
                )
            continue

        reply = response.content or ""
        break
    else:
        # loop exhausted while still calling tools -> don't spin forever
        reply = _FALLBACK_REPLY

    latency_ms = int((time.perf_counter() - start) * 1000)
    return AgentResult(
        reply=reply,
        steps=steps,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )
