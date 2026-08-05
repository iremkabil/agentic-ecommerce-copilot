"""FAQ/policy retrieval tool: RAG over faq.md + policies.md.

The two files are chunked by heading, embedded, and searched. Every returned
passage carries its ``source`` ("faq" or "policies") and ``title`` so the agent
can ground answers and, later, so the output guardrail can verify that a policy
claim actually came from ``policies.md``.
"""

from __future__ import annotations

from pathlib import Path

from copilot.retrieval.chunking import split_markdown_sections
from copilot.retrieval.embed import Embedder
from copilot.retrieval.index import Document, VectorIndex


def build_faq_index(
    embedder: Embedder,
    faq_path: str | Path,
    policies_path: str | Path,
) -> VectorIndex:
    """Chunk faq.md (### sections) and policies.md (## sections) into one index."""
    index = VectorIndex(embedder)
    docs: list[Document] = []

    faq_text = Path(faq_path).read_text(encoding="utf-8")
    for title, body in split_markdown_sections(faq_text, "### "):
        docs.append(
            Document(
                id=f"faq::{title}",
                text=f"{title}\n{body}",
                metadata={"source": "faq", "title": title},
            )
        )

    policy_text = Path(policies_path).read_text(encoding="utf-8")
    for title, body in split_markdown_sections(policy_text, "## "):
        docs.append(
            Document(
                id=f"policy::{title}",
                text=f"{title}\n{body}",
                metadata={"source": "policies", "title": title},
            )
        )

    index.add(docs)
    return index


def faq_retrieval(index: VectorIndex, query: str, k: int = 3) -> list[dict]:
    """Return the top-k FAQ/policy passages for a query."""
    hits = index.search(query, k=k)
    return [
        {
            "source": doc.metadata["source"],
            "title": doc.metadata["title"],
            "text": doc.text,
            "score": round(score, 4),
        }
        for doc, score in hits
    ]
