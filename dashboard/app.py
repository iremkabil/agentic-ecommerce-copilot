"""Streamlit admin dashboard: conversations, intents, tool usage, handoffs,
guardrail events, eval metrics + trend. [Day 12]

Reads directly from the configured DB, the same way eval/run_eval.py does --
an admin dashboard is an internal ops view, not a client of the public API
(see PROJECT_PLAN.md §3's architecture diagram).

Chart color follows a job-based rule, not taste: intent/tool counts are
nominal categories with no natural order, so every bar gets one sequential
hue (a value-ramp per bar would double-encode length as color and imply an
order that isn't there). The guardrail action breakdown is genuine status
data -- allow/block/escalate really do mean good/warning/critical -- so it
gets the reserved status palette instead, always paired with the axis's text
labels rather than color alone. The eval-metric trend uses the fixed
categorical order (never re-cycled) capped at 4 series, per the same
palette's safety ceiling. All hex values are the validated defaults from the
project's dataviz reference palette.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from copilot.config import get_settings
from copilot.db.session import SessionLocal
from dashboard.queries import (
    conversations_table,
    eval_run_history,
    guardrail_action_breakdown,
    guardrail_events_table,
    handoff_queue,
    intent_distribution,
    kpi_summary,
    latest_eval_run,
    messages_for_conversation,
    tool_usage,
)

st.set_page_config(page_title="Paperbloom Copilot — Admin", page_icon="📊", layout="wide")

settings = get_settings()

_SEQUENTIAL_BLUE = "#2a78d6"  # magnitude bars: one hue, no natural order to rank
_STATUS = {"allow": "#0ca30c", "block": "#fab219", "escalate": "#d03b3b"}
_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, capped at 4
_NON_METRIC_COLUMNS = {"run_name", "created_at", "intent_per_label_f1"}


@st.cache_data(ttl=10)
def _load_dashboard_data() -> dict:
    with SessionLocal() as session:
        return {
            "kpi": kpi_summary(session),
            "conversations": conversations_table(session),
            "intents": intent_distribution(session),
            "tools": tool_usage(session),
            "handoffs": handoff_queue(session),
            "guardrail_breakdown": guardrail_action_breakdown(session),
            "guardrail_events": guardrail_events_table(session),
            "latest_run": latest_eval_run(session),
            "eval_history": eval_run_history(session),
        }


def _load_messages(conversation_id: str) -> pd.DataFrame:
    with SessionLocal() as session:
        return messages_for_conversation(session, conversation_id)


def _magnitude_bar_chart(df: pd.DataFrame, category_col: str, value_col: str) -> alt.Chart:
    """One sequential hue, sorted by magnitude -- the default form for
    comparing nominal categories with no inherent order."""
    return (
        alt.Chart(df)
        .mark_bar(color=_SEQUENTIAL_BLUE, cornerRadius=2)
        .encode(
            x=alt.X(f"{value_col}:Q", title="Count"),
            y=alt.Y(f"{category_col}:N", sort="-x", title=None),
            tooltip=[category_col, value_col],
        )
    )


with st.sidebar:
    st.header("Settings")
    st.caption(f"Database: `{settings.database_url}`")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

st.title("📊 Paperbloom Copilot — Admin Dashboard")
st.caption("Synthetic-data demo. No real brand, customer, or order data.")

data = _load_dashboard_data()

# --- KPI row ---------------------------------------------------------------

kpi = data["kpi"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversations", kpi["conversations"])
col2.metric("Messages", kpi["messages"])
col3.metric("Handoff rate", f"{kpi['handoff_rate']:.1%}")
col4.metric(
    "Avg latency", f"{kpi['avg_latency_ms']:.0f} ms" if kpi["avg_latency_ms"] is not None else "—"
)

st.divider()

# --- Conversations -----------------------------------------------------------

st.subheader("Conversations")
conversations = data["conversations"]
if conversations.empty:
    st.info(
        "No conversations yet. Chat with the agent "
        "(`streamlit run dashboard/chat.py`) to populate this view."
    )
else:
    st.dataframe(conversations, width="stretch", hide_index=True)
    selected = st.selectbox("View transcript for conversation:", conversations["conversation_id"])
    if selected:
        st.dataframe(_load_messages(selected), width="stretch", hide_index=True)

st.divider()

# --- Intent distribution / Tool usage -----------------------------------------

col_intent, col_tools = st.columns(2)
with col_intent:
    st.subheader("Intent distribution")
    intents = data["intents"]
    if intents.empty:
        st.info("No classified messages yet.")
    else:
        st.altair_chart(_magnitude_bar_chart(intents, "intent", "count"), width="stretch")
        with st.expander("Show as table"):
            st.dataframe(intents, width="stretch", hide_index=True)

with col_tools:
    st.subheader("Tool usage")
    tools = data["tools"]
    if tools.empty:
        st.info("No tool calls yet.")
    else:
        st.altair_chart(_magnitude_bar_chart(tools, "tool_name", "count"), width="stretch")
        with st.expander("Show as table"):
            st.dataframe(tools, width="stretch", hide_index=True)

st.divider()

# --- Handoffs ------------------------------------------------------------------

st.subheader("Handoff queue")
handoffs = data["handoffs"]
if handoffs.empty:
    st.info("No handoffs yet.")
else:
    open_count = int((handoffs["status"] == "open").sum())
    st.caption(f"{open_count} open of {len(handoffs)} total")
    st.dataframe(handoffs, width="stretch", hide_index=True)

st.divider()

# --- Guardrail events ------------------------------------------------------------

st.subheader("Guardrail events")
breakdown = data["guardrail_breakdown"]
if breakdown.empty:
    st.info("No guardrail events yet.")
else:
    # status data, not identity -- allow/block/escalate really do mean
    # good/warning/critical, so this is the one chart that earns the status
    # palette. Text labels on the axis mean color never carries meaning alone.
    action_order = [a for a in ("allow", "block", "escalate") if a in breakdown["action"].values]
    chart = (
        alt.Chart(breakdown)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("action:N", title=None, sort=action_order),
            y=alt.Y("count:Q", title="Events"),
            color=alt.Color(
                "action:N",
                scale=alt.Scale(domain=list(_STATUS.keys()), range=list(_STATUS.values())),
                legend=alt.Legend(title="Action"),
            ),
            tooltip=["action", "count"],
        )
    )
    st.altair_chart(chart, width="stretch")
    with st.expander("Recent guardrail events"):
        st.dataframe(data["guardrail_events"], width="stretch", hide_index=True)

st.divider()

# --- Eval metrics ----------------------------------------------------------------

st.subheader("Evaluation metrics")
latest_run = data["latest_run"]
if latest_run is None:
    st.info("No eval runs yet. Run `python -m eval.run_eval` to populate this view.")
else:
    m = latest_run.metrics or {}
    st.caption(
        f"Latest run: **{latest_run.run_name}** "
        f"({latest_run.created_at:%Y-%m-%d %H:%M} UTC), {m.get('n_cases', '?')} cases"
    )
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Intent accuracy", f"{m.get('intent_accuracy', 0):.1%}")
    mcol2.metric("Tool micro-F1", f"{m.get('tool_micro_f1', 0):.1%}")
    mcol3.metric("Guardrail block rate", f"{m.get('guardrail_block_rate', 0):.1%}")
    mcol4.metric("Handoff recall", f"{m.get('handoff_recall', 0):.1%}")
    with st.expander("All metrics for this run"):
        st.json(m)

    history = data["eval_history"]
    if len(history) > 1:
        st.markdown("**Trend across runs**")
        metric_options = [c for c in history.columns if c not in _NON_METRIC_COLUMNS]
        default_metrics = [
            c
            for c in ("intent_accuracy", "tool_micro_f1", "order_completion_rate")
            if c in metric_options
        ]
        chosen = st.multiselect(
            "Metrics to plot",
            metric_options,
            default=default_metrics or metric_options[:3],
            max_selections=4,
        )
        if chosen:
            long_df = history.melt(
                id_vars=["run_name"], value_vars=chosen, var_name="metric", value_name="value"
            )
            trend_chart = (
                alt.Chart(long_df)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("run_name:N", title="Run", sort=list(history["run_name"])),
                    y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color(
                        "metric:N",
                        scale=alt.Scale(range=_CATEGORICAL[: len(chosen)]),
                        legend=alt.Legend(title="Metric"),
                    ),
                    tooltip=["run_name", "metric", "value"],
                )
            )
            st.altair_chart(trend_chart, width="stretch")
