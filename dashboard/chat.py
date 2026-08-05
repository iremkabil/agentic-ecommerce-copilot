"""Streamlit demo chat UI calling the FastAPI /chat endpoint. [Day 6]

A thin client: no agent logic lives here. It just renders the conversation,
posts each new user message to ``POST /chat``, and surfaces the returned
``steps`` (the tool calls the agent made) so the loop is visible for demos
and debugging, not just the final reply.
"""

from __future__ import annotations

import httpx
import streamlit as st

from copilot.config import get_settings

st.set_page_config(page_title="Paperbloom Copilot — Demo Chat", page_icon="🌸")

settings = get_settings()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict]: role, content, steps
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

with st.sidebar:
    st.header("Settings")
    api_base_url = st.text_input("API base URL", value=settings.api_base_url).rstrip("/")
    st.caption(f"conversation_id: `{st.session_state.conversation_id or '(new)'}`")
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

st.title("🌸 Paperbloom Copilot — Demo Chat")
st.caption("Synthetic-data demo. No real brand, customer, or order data.")


def _render_steps(steps: list[dict]) -> None:
    """Show the agent's tool calls for a turn, if any, in a collapsed expander."""
    if not steps:
        return
    with st.expander(f"Tool steps ({len(steps)})"):
        for i, step in enumerate(steps, start=1):
            st.markdown(f"**{i}. `{step['tool_name']}`**")
            st.json(step["tool_input"], expanded=False)
            st.json(step["tool_output"], expanded=False)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_steps(message.get("steps", []))

if user_message := st.chat_input("Ask about products, orders, shipping..."):
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        try:
            response = httpx.post(
                f"{api_base_url}/chat",
                json={
                    "message": user_message,
                    "conversation_id": st.session_state.conversation_id,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the copilot API at {api_base_url}: {exc}")
        else:
            st.session_state.conversation_id = body["conversation_id"]
            st.markdown(body["reply"])
            _render_steps(body["steps"])
            st.session_state.messages.append(
                {"role": "assistant", "content": body["reply"], "steps": body["steps"]}
            )
