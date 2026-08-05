"""Offline test for the Streamlit demo chat UI (Day 6).

Drives dashboard/chat.py with streamlit's AppTest harness. ``httpx.post`` is
monkeypatched to a canned response so the test never touches the network or
requires a running API, matching the project's offline-testing convention.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

CHAT_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "chat.py")


def _fake_post(url, json, timeout):  # noqa: ARG001 - signature mirrors httpx.post
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "conversation_id": "conv-123",
            "reply": "The Bamboo Desk Organizer is $29.00.",
            "steps": [
                {
                    "tool_name": "product_search",
                    "tool_input": {"query": "bamboo organizer"},
                    "tool_output": {"results": ["Bamboo Desk Organizer"]},
                }
            ],
        },
    )


def test_chat_ui_sends_message_and_renders_reply_with_steps(monkeypatch):
    monkeypatch.setattr("httpx.post", _fake_post)

    at = AppTest.from_file(CHAT_PATH)
    at.run()
    assert not at.exception

    at.chat_input[0].set_value("do you have a bamboo organizer?").run()
    assert not at.exception

    assert at.session_state.conversation_id == "conv-123"
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "The Bamboo Desk Organizer is $29.00." in markdown_text
    assert len(at.expander) == 1
    assert "Tool steps (1)" in at.expander[0].label


def test_chat_ui_shows_error_on_request_failure(monkeypatch):
    import httpx

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("httpx.post", _raise)

    at = AppTest.from_file(CHAT_PATH)
    at.run()
    at.chat_input[0].set_value("hello").run()

    assert not at.exception
    assert len(at.error) == 1
    assert "Could not reach the copilot API" in at.error[0].value
