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

🚧 In development. **✅ MVP complete (Day 5).** A working agent answers grounded
questions end to end via `POST /chat`.
- **Day 1:** scaffold, configuration, Docker, health check.
- **Day 2:** SQLAlchemy models (10 tables), DB session, seed script, synthetic data.
- **Day 3:** retrieval layer (pluggable embedder + vector index) and product/FAQ tools.
- **Day 4:** deterministic `shipping_calculator`, `get_order_status`, and the `llm/` interface.
- **Day 5:** the agent loop (native tool calling), system prompt, tool registry, and the
  `POST /chat` endpoint with full conversation + tool + telemetry logging. **MVP done.**
- **Day 6:** Streamlit demo chat UI (`dashboard/chat.py`) — talks to `POST /chat`, keeps
  `conversation_id` in session state across turns, and shows each turn's tool calls in an
  expander. API base URL is configurable (`COPILOT_API_BASE_URL`, editable in the sidebar).

Try it:

```bash
python -m copilot.db.seed --reset          # seed the demo data
uvicorn copilot.api.main:app --reload      # needs a running LLM (Ollama or hosted)
# POST http://localhost:8000/chat  {"message": "do you offer gift wrapping?"}

streamlit run dashboard/chat.py            # demo chat UI, talks to the API above
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
# Dashboard -> http://localhost:8501   (placeholder until Day 12)
```

## Tech stack

Python 3.11 · FastAPI · Streamlit · Pydantic v2 · SQLAlchemy (SQLite / Postgres) ·
sentence-transformers + FAISS · Docker.

## License

MIT — see [LICENSE](./LICENSE).
