# Agentic E-commerce Support & Sales Copilot

An LLM-powered customer-support and sales **agent** (not a scripted chatbot) for a
fictional store. It classifies intent, calls tools, retrieves product/policy knowledge,
drafts orders, enforces guardrails, and escalates to a human when needed.

> ⚠️ **Synthetic-data & ethics notice.** This project uses **100% synthetic, fabricated
> data**. There is no real brand, customer, order, or personal data anywhere in this
> repository. The brand "Paperbloom" is invented for demonstration only, and the agent
> never charges a real payment method.

See **[PROJECT_PLAN.md](./PROJECT_PLAN.md)** for the full design, roadmap, and evaluation plan.

---

## Status

🚧 In development, following the 14-day build timeline in
[PROJECT_PLAN.md §11](./PROJECT_PLAN.md#11-development-timeline-1014-days) (full V1/V2 scope in
[§10 Roadmap](./PROJECT_PLAN.md#10-roadmap-mvp--v1--v2)). **✅ MVP complete (Day 5).** Currently
on **Day 12 of 14** — Days 13–14 (tests/tooling polish, README/demo polish) remain. A
working agent answers grounded questions end to end via `POST /chat`.
- **Day 1:** scaffold, configuration, Docker, health check.
- **Day 2:** SQLAlchemy models (10 tables), DB session, seed script, synthetic data.
- **Day 3:** retrieval layer (pluggable embedder + vector index) and product/FAQ tools.
- **Day 4:** deterministic `shipping_calculator`, `get_order_status`, and the `llm/` interface.
- **Day 5:** the agent loop (native tool calling), system prompt, tool registry, and the
  `POST /chat` endpoint with full conversation + tool + telemetry logging. **MVP done.**
- **Day 6:** Streamlit demo chat UI (`dashboard/chat.py`) — talks to `POST /chat`, keeps
  `conversation_id` in session state across turns, and shows each turn's tool calls in an
  expander. API base URL is configurable (`COPILOT_API_BASE_URL`, editable in the sidebar).
- **Day 7:** intent classifier (`agent/intent.py`) — a single few-shot LLM call returns a
  label + confidence for every user message ahead of the tool-calling loop; low-confidence
  predictions are flagged explicitly (`COPILOT_INTENT_CONFIDENCE_THRESHOLD`) for Day 9's
  handoff logic. Wired into `handle_chat`, so every user message is logged with its
  `intent`/`intent_confidence` in the `messages` table.
- **Day 8:** order flow — `OrderDraft` schema (`agent/schemas.py`) plus three new tools
  (`tools/orders.py`): `extract_order_fields`, `detect_missing_fields`, `create_order_draft`.
  Slot filling needs no special-case state machine — the model re-states what it already
  knows from its own conversation history each turn; only `create_order_draft` touches the
  DB, persisting an `Order` + `OrderItem`s + `Customer` once every required field is present.
- **Day 9:** rule-based input/output guardrails (`guardrails/input_rules.py`,
  `output_rules.py`) and an escalation matrix (`guardrails/escalation.py`) wired into
  `handle_chat`: jailbreak/PII/abuse/out-of-scope messages are blocked or escalated before
  the classifier or orchestrator ever run; drafted replies are checked for ungrounded
  prices/policy promises and prompt/tool-name leakage before they ship. `tools/handoff.py`
  persists `HandoffCase`s (and marks the conversation `handed_off`) both automatically
  (explicit human request, low-confidence intent, escalating input rules) and via a
  `human_handoff` tool the model can call itself for complaints/uncovered policy questions.
  Every guardrail decision is logged to `guardrail_events`.
- **Day 10:** evaluation harness (metrics 1-3) — `data/test_cases.csv` (71 hand-authored,
  grounded rows across all 9 intents plus jailbreak/PII/abuse/prohibited-advice adversarial
  cases) + `eval/metrics.py` (intent accuracy/macro-F1, tool-selection precision/recall/
  micro-F1, order completion rate — all pure functions) + `eval/run_eval.py`, which drives
  the *live* `handle_chat` pipeline per row and prints a summary table
  (`python -m eval.run_eval`).
- **Day 11:** metrics 4-6 — missing-field detection precision/recall (a new
  `expected_missing_fields` CSV column, validated offline against `detect_missing_fields`
  for every place_order case), guardrail block rate / false-positive rate (benign vs.
  adversarial, via `evaluate_input` — still 0 mismatches across all 71 rows), and handoff
  precision/recall (via a real `HandoffCase` DB lookup per case). `eval/run_eval.py` now
  persists every run: one `eval_runs` row (all 6 metrics as JSON) + one `eval_results` row
  per case (`python -m eval.run_eval --run-name my-run`).
- **Day 12:** Streamlit admin dashboard (`dashboard/app.py`) — conversations (with per-
  transcript drill-down), intent distribution, tool usage, the handoff queue, guardrail
  events, and the latest eval run's metrics with a trend chart across runs. Reads the DB
  directly (`dashboard/queries.py`, a pure query layer, unit-tested the same way as every
  other DB-touching module). Chart color follows a job-based rule: intent/tool counts are
  nominal categories so they get one sequential hue; the guardrail action breakdown is real
  status data (allow/block/escalate) so it gets the reserved status palette instead.

Try it:

```bash
python -m copilot.db.seed --reset          # seed the demo data
uvicorn copilot.api.main:app --reload      # needs a running LLM (Ollama or hosted)
# POST http://localhost:8000/chat  {"message": "do you offer gift wrapping?"}

streamlit run dashboard/chat.py            # demo chat UI, talks to the API above
streamlit run dashboard/app.py             # admin dashboard, reads the DB directly
```

## Quickstart (local)

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install the package (with retrieval + dev extras)
pip install -e ".[retrieval,dev]"

# 3. configure
cp .env.example .env

# 4. run the API
uvicorn copilot.api.main:app --reload
# -> open http://localhost:8000/health  and  http://localhost:8000/docs

# 5. run the tests
pytest
```

## Quickstart (Docker)

```bash
cp .env.example .env
docker compose up --build
# API       -> http://localhost:8000/health
# Dashboard -> http://localhost:8501   (admin dashboard: dashboard/app.py)
```

## Tech stack

Python 3.11 · FastAPI · Streamlit · Pydantic v2 · SQLAlchemy (SQLite / Postgres) ·
sentence-transformers + FAISS · Docker.

## License

MIT — see [LICENSE](./LICENSE).
