"""Tests for the LLM abstraction: offline scripted client + wire (de)serialization.

No network is used: we test the pure serialize/parse helpers and the scripted
client that the Day 5 agent loop will run against. OpenAICompatibleClient's
error handling (Day 13) is tested with httpx.post monkeypatched, same
technique as dashboard/chat.py's tests -- still zero real network.
"""

from __future__ import annotations

import httpx
import pytest

from copilot.llm.base import ChatMessage, LLMProviderError, LLMResponse, ToolCall, ToolSpec
from copilot.llm.providers import (
    OpenAICompatibleClient,
    ScriptedLLMClient,
    messages_to_wire,
    parse_response,
    tools_to_wire,
)


def test_scripted_client_replays_in_order_and_records_calls():
    client = ScriptedLLMClient(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="1", name="product_search", arguments={"query": "pen"})]
            ),
            LLMResponse(content="Here are some pens."),
        ]
    )
    first = client.chat([ChatMessage(role="user", content="find a pen")])
    assert first.tool_calls[0].name == "product_search"
    second = client.chat([ChatMessage(role="user", content="thanks")])
    assert second.content == "Here are some pens."
    assert len(client.calls) == 2  # both calls recorded


def test_messages_to_wire_serializes_tool_flow():
    messages = [
        ChatMessage(role="system", content="you are helpful"),
        ChatMessage(role="user", content="find a pen"),
        ChatMessage(
            role="assistant", tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "pen"})]
        ),
        ChatMessage(role="tool", tool_call_id="c1", name="search", content='{"results": []}'),
    ]
    wire = messages_to_wire(messages)
    # assistant tool call is serialized with a JSON-string arguments field
    assert wire[2]["tool_calls"][0]["function"]["name"] == "search"
    assert wire[2]["tool_calls"][0]["function"]["arguments"] == '{"q": "pen"}'
    # tool result carries the matching tool_call_id
    assert wire[3]["tool_call_id"] == "c1"


def test_tools_to_wire_shape():
    spec = ToolSpec(
        name="shipping_calculator",
        description="compute shipping",
        parameters={"type": "object", "properties": {"country": {"type": "string"}}},
    )
    wire = tools_to_wire([spec])
    assert wire[0]["type"] == "function"
    assert wire[0]["function"]["name"] == "shipping_calculator"


def test_parse_response_extracts_content_tool_calls_and_usage():
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "product_search",
                                "arguments": '{"query": "notebook"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }
    resp = parse_response(data)
    assert resp.content is None
    assert resp.tool_calls[0].name == "product_search"
    assert resp.tool_calls[0].arguments == {"query": "notebook"}
    assert resp.usage.prompt_tokens == 12


def test_parse_response_handles_malformed_arguments():
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "f", "arguments": "not-json"},
                        }
                    ],
                }
            }
        ]
    }
    resp = parse_response(data)
    assert resp.tool_calls[0].arguments == {}  # falls back to empty dict, no crash


# --- OpenAICompatibleClient error handling (Day 13) --------------------------


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url="http://localhost:11434/v1", model="llama3.1")


def test_openai_compatible_client_wraps_connection_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("httpx.post", _raise)

    with pytest.raises(LLMProviderError):
        _client().chat([ChatMessage(role="user", content="hi")])


def test_openai_compatible_client_wraps_timeout(monkeypatch):
    def _raise(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("httpx.post", _raise)

    with pytest.raises(LLMProviderError):
        _client().chat([ChatMessage(role="user", content="hi")])


def test_openai_compatible_client_wraps_non_2xx_status(monkeypatch):
    def _fake_post(*args, **kwargs):
        request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
        return httpx.Response(500, request=request, json={"error": "internal"})

    monkeypatch.setattr("httpx.post", _fake_post)

    with pytest.raises(LLMProviderError):
        _client().chat([ChatMessage(role="user", content="hi")])


def test_openai_compatible_client_succeeds_on_2xx(monkeypatch):
    def _fake_post(*args, **kwargs):
        request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
        body = {"choices": [{"message": {"content": "hello!"}}], "usage": {}}
        return httpx.Response(200, request=request, json=body)

    monkeypatch.setattr("httpx.post", _fake_post)

    response = _client().chat([ChatMessage(role="user", content="hi")])
    assert response.content == "hello!"
