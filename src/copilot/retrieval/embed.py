"""Text embedding, behind a small interface so the rest of the code never
depends on a concrete model.

Two implementations satisfy the ``Embedder`` protocol:

* ``SentenceTransformerEmbedder`` - the real, semantic embeddings used in normal
  operation. It lazily imports ``sentence_transformers`` so simply importing this
  module never requires torch to be installed.
* ``HashingEmbedder`` - a deterministic, dependency-light fallback using the
  feature-hashing ("hashing trick") technique. It needs no model download, so it
  keeps the retrieval layer fully testable offline and in CI. It is *lexical*,
  not semantic: it's for the pipeline and tests, not for production quality.

Both return L2-normalized ``float32`` arrays, so cosine similarity reduces to a
plain dot product downstream.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Anything that turns a list of strings into a (N, dim) float32 matrix."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder (no external model).

    Each token is hashed to a bucket in ``[0, dim)`` with a signed contribution,
    then the vector is L2-normalized. Same text -> same vector, always.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _tokenize(text):
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                bucket = h % self.dim
                sign = 1.0 if (h >> 8) & 1 == 0 else -1.0
                vecs[i, bucket] += sign
            norm = float(np.linalg.norm(vecs[i]))
            if norm > 0.0:
                vecs[i] /= norm
        return vecs


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers (lazy import)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: avoids torch at import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        embs = self._model.encode(
            texts,
            normalize_embeddings=True,  # so cosine == dot product
            convert_to_numpy=True,
        )
        return embs.astype("float32")


def get_embedder(settings=None) -> Embedder:
    """Build the embedder chosen by settings.embedding_backend."""
    from copilot.config import get_settings

    s = settings or get_settings()
    if s.embedding_backend == "hashing":
        return HashingEmbedder(dim=s.embedding_dim)
    return SentenceTransformerEmbedder(s.embedding_model)
