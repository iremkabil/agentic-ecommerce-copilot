# CLAUDE.md — Working Guide for This Repository

This file is auto-loaded by Claude Code as project memory. Read it first, then
read `PROJECT_PLAN.md` before doing anything.

## What this project is
An **Agentic E-commerce Support & Sales Copilot** for a *fictional* stationery
store ("Paperbloom"), built on **100% synthetic data** (no real brand, customer,
order, or PII). It is a portfolio project demonstrating an LLM agent with intent
detection, tool calling, RAG, order drafting, guardrails, human handoff, an admin
dashboard, and an evaluation harness.

## Source of truth
`PROJECT_PLAN.md` is authoritative for scope, architecture, data model, and the
build timeline. In particular:
- §10 Roadmap (MVP / V1 / V2)
- §11 Development Timeline (the day-by-day plan you are executing)

## Current state
**MVP complete (Day 5).** `pytest` is green (43 passed as of Day 5). Do not break
existing tests.

Implemented so far:
- `config.py` — typed settings via pydantic-settings (`COPILOT_` env prefix).
- `db/` — SQLAlchemy 2.0 models (10 tables), session, seed script.
- `retrieval/` — `Embedder` protocol (sentence-transformers **or** offline
  hashing fallback) + numpy `VectorIndex` + markdown chunking.
- `tools/` — `product_search`, `get_product_details`, `faq_retrieval`,
  `shipping_calculator`, `get_order_status`.
- `llm/` — provider-agnostic `LLMClient` (OpenAI-compatible client for
  Ollama/hosted + a `ScriptedLLMClient` for offline tests).
- `agent/` — system prompt (the "constitution"), tool registry, the native
  tool-calling loop (`orchestrator.py`), and `service.handle_chat` with logging.
- `api/` — `GET /health` and `POST /chat`.

## Remaining timeline (do these IN ORDER, one at a time)
- **Day 6** — Streamlit demo chat UI (`dashboard/chat.py`) that calls `POST /chat`,
  keeps `conversation_id` in session state, and shows the agent's tool steps.
- **Day 7** — Intent classifier (`agent/intent.py`): message → (label, confidence),
  logged per message.
- **Day 8** — Order flow: `OrderDraft` schema + `tools/orders.py`
  (`extract_order_fields`, `detect_missing_fields`, `create_order_draft`).
- **Day 9** — Guardrails (`guardrails/input_rules.py`, `output_rules.py`) +
  `tools/handoff.py` + escalation matrix + `guardrail_events` / `handoff_cases`.
- **Day 10-11** — Eval: `data/test_cases.csv` + `eval/run_eval.py` + `eval/metrics.py`
  (intent accuracy/F1, tool-selection F1, order completion, missing-field, guardrail
  block & false-positive rates, handoff accuracy), persisted to `eval_runs`/`eval_results`.
- **Day 12** — Streamlit **admin dashboard** (`dashboard/app.py`): conversations,
  intents, tool usage, handoffs, guardrail events, latest eval metrics + trend.
- **Day 13** — Tests, ruff/black/pre-commit, docstrings, error handling.
- **Day 14** — README polish, architecture diagram, screenshots, resume/LinkedIn copy.

## Conventions you MUST follow (they are already established in the code)
1. **Dependency inversion.** New external capabilities go behind a small protocol
   with swappable implementations (see `Embedder`, `LLMClient`). Don't hard-wire a
   concrete library into business logic.
2. **Pure, testable cores.** Tools are pure functions that take their dependencies
   (session, index) explicitly; executors in `agent/registry.py` are the defensive
   LLM-facing wrappers. Keep this split.
3. **Everything is offline-testable.** Tests must NOT hit the network, download a
   model, or require a running LLM. Use `HashingEmbedder` and `ScriptedLLMClient`.
   For FastAPI + in-memory SQLite tests, use `StaticPool` + `check_same_thread=False`.
4. **Keep `pytest` green** and add tests for every new non-trivial piece.
5. **SQLAlchemy 2.0 typed style** (`Mapped`, `mapped_column`), JSON columns for
   flexible blobs, `session_scope` for writes.
6. **Docstrings explain _why_, not just what.** Match the existing tone.
7. **No unnecessary dependencies or enterprise complexity.** Prefer the simplest
   thing that is correct at this scale (e.g., numpy exact search over FAISS).
8. **Synthetic-data ethics** stay explicit; never introduce real brands/PII.
9. Respect `ruff`/`black` config in `pyproject.toml` (line length 100).

## Per-phase workflow (repeat for each day)
1. State a short plan for the day (what files, what design choices, and why).
2. Implement it.
3. Run `pytest` and make it green; add/extend tests.
4. Update the **Status** section in `README.md` for the completed day.
5. Give a brief summary (what changed + the reasoning/tradeoffs).
6. **STOP and wait for my explicit approval before starting the next day.**
   Suggest a git commit message, but do not jump ahead.

## Commands (Windows PowerShell; adjust the venv path if different)
```powershell
# activate the venv
.\.venv\Scripts\Activate.ps1

# install (once, if needed)
pip install -e ".[retrieval,dev]"

# seed the demo database
python -m copilot.db.seed --reset

# run the tests (must stay green)
pytest -q

# run the API (needs a running LLM: Ollama or a hosted OpenAI-compatible endpoint)
uvicorn copilot.api.main:app --reload
# docs at http://localhost:8000/docs

# run the Streamlit chat UI (Day 6+)
streamlit run dashboard/chat.py
```

## Notes
- `pyproject.toml` sets `pythonpath = ["src"]`, so `pytest` and imports work from
  the repo root.
- Tests run with the offline hashing backend regardless of `.env`; real semantic
  search uses `COPILOT_EMBEDDING_BACKEND=sentence-transformers`.
