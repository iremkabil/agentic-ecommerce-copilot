"""Application configuration.

All runtime configuration is centralized here and loaded from environment
variables (and an optional local ``.env`` file) via ``pydantic-settings``.

Why one typed Settings object instead of reading ``os.environ`` everywhere:
- every module imports the *same* validated config, so a typo fails once, loudly;
- types are enforced (e.g. ``max_agent_steps`` is guaranteed to be an int);
- swapping SQLite -> Postgres, or Ollama -> a hosted LLM, is a config change,
  not a code change.

Environment variables are prefixed with ``COPILOT_`` (see ``.env.example``) so
they never collide with unrelated variables on the host.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Agentic E-commerce Support & Sales Copilot"
    environment: str = Field(default="development")  # development | production
    log_level: str = Field(default="INFO")

    # --- Paths ---
    data_dir: Path = Field(default=Path("data"))
    faiss_index_dir: Path = Field(default=Path(".faiss"))

    # --- Database ---
    # Default: a local SQLite file (zero-config, so anyone can clone and run).
    # Swap in a Postgres URL to scale up, e.g.:
    #   postgresql+psycopg://user:pass@localhost:5432/copilot
    database_url: str = Field(default="sqlite:///./copilot.db")

    # --- LLM provider (kept generic + swappable) ---
    # Use a local model via Ollama for $0 dev cost, or point at any
    # OpenAI-compatible / hosted endpoint. Nothing else in the codebase
    # should know which provider is behind this.
    llm_provider: str = Field(default="ollama")
    llm_model: str = Field(default="llama3.1")
    llm_base_url: str | None = Field(default="http://localhost:11434/v1")
    llm_api_key: str | None = Field(default=None)
    llm_temperature: float = Field(default=0.0)

    # --- Retrieval ---
    # embedding_backend selects the implementation behind the Embedder protocol:
    #   "sentence-transformers" -> real semantic embeddings (needs the model)
    #   "hashing"               -> dependency-light, offline fallback (tests/CI)
    embedding_backend: str = Field(default="sentence-transformers")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=256)  # used by the hashing fallback
    # retrieval_backend selects the vector store: "numpy" (exact, always available)
    # or "faiss" (optional accelerator for larger corpora).
    retrieval_backend: str = Field(default="numpy")

    # --- Agent ---
    max_agent_steps: int = Field(default=6)
    # Below this, IntentResult.low_confidence is True (drives Day 9 handoff bias).
    intent_confidence_threshold: float = Field(default=0.5)

    # --- Dashboard/client ---
    # Base URL the Streamlit demo chat (dashboard/chat.py) uses to reach the
    # FastAPI /chat endpoint. Separate process, so it needs its own address
    # even when everything runs on one machine in dev.
    api_base_url: str = Field(default="http://localhost:8000")


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance.

    ``lru_cache`` means the environment is parsed exactly once; every caller
    gets the same object instead of re-reading files on each import.
    """
    return Settings()
