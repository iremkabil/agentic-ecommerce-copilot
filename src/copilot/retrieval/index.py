"""A tiny vector index with exact cosine search.

Why exact numpy search (and not FAISS by default)
-------------------------------------------------
For a catalog of dozens of products and ~20 FAQ/policy chunks, an exact
brute-force cosine search over a normalized matrix is *simpler, correct, and
actually faster* than building and querying a FAISS index (whose approximate
structures pay off only from ~10k+ vectors). FAISS is wired as a config option
(``retrieval_backend=faiss``) to demonstrate the skill and to be ready for
scale, but numpy is the honest default here.

Because embeddings are L2-normalized, cosine similarity is just a dot product,
so search is a single matrix-vector multiply plus an argsort.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from copilot.retrieval.embed import Embedder


@dataclass
class Document:
    """A searchable unit: an id, the text that gets embedded, and metadata used
    for filtering and for building tool responses."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


class VectorIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.documents: list[Document] = []
        self._matrix: np.ndarray | None = None  # (N, dim), rows L2-normalized

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        embs = self.embedder.encode([d.text for d in docs])
        self.documents.extend(docs)
        self._matrix = embs if self._matrix is None else np.vstack([self._matrix, embs])

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Callable[[Document], bool] | None = None,
    ) -> list[tuple[Document, float]]:
        """Return up to ``k`` (document, score) pairs ranked by cosine similarity.

        ``filter_fn`` is applied as a post-filter (fine at this scale; at large
        scale you would pre-filter or use metadata-aware indexes).
        """
        if self._matrix is None or not self.documents:
            return []
        q = self.embedder.encode([query])[0]  # normalized
        scores = self._matrix @ q  # cosine, since rows are normalized
        results: list[tuple[Document, float]] = []
        for idx in np.argsort(-scores):
            doc = self.documents[int(idx)]
            if filter_fn is not None and not filter_fn(doc):
                continue
            results.append((doc, float(scores[int(idx)])))
            if len(results) >= k:
                break
        return results

    # --- persistence ---------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.save(directory / "matrix.npy", self._matrix)
        (directory / "documents.json").write_text(
            json.dumps([asdict(d) for d in self.documents]), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> VectorIndex:
        directory = Path(directory)
        index = cls(embedder)
        docs = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        index.documents = [Document(**d) for d in docs]
        matrix_path = directory / "matrix.npy"
        if matrix_path.exists():
            index._matrix = np.load(matrix_path)
        return index
