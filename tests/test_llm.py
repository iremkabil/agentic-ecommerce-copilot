"""Tests for the LLM abstraction: offline scripted client + wire (de)serialization.

No network is used: we test the pure serialize/parse helpers and the scripted
client that the Day 5 agent loop will run against.
"""

from __future__ import annotations

from copilot.llm.base import ChatMessage, LLMResponse, ToolCall, ToolSpec
from copilot.llm.providers import (
    ScriptedLLMClient,
    messages_to_wire,
    parse_response,
    tools_to_wire,
)


def test_scripted_client_replays_in_order_and_records_calls():
    client = ScriptedLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="product_search", arguments={"query": "pen"})]),
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
        ChatMessage(role="assistant", tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "pen"})]),
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
                            "function": {"name": "product_search", "arguments": '{"query": "notebook"}'},
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
    data = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "f", "arguments": "not-json"}}
    ]}}]}
    resp = parse_response(data)
    assert resp.tool_calls[0].arguments == {}  # falls back to empty dict, no crash
