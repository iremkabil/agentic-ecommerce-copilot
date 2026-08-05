"""Tests for the retrieval layer (embedder + vector index + chunking).

All offline: they use HashingEmbedder, so no model download is required.
"""

from __future__ import annotations

import numpy as np

from copilot.retrieval.chunking import split_markdown_sections
from copilot.retrieval.embed import HashingEmbedder
from copilot.retrieval.index import Document, VectorIndex


def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(dim=128)
    v1 = e.encode(["bamboo desk organizer"])
    v2 = e.encode(["bamboo desk organizer"])
    assert v1.shape == (1, 128)
    assert np.allclose(v1, v2)  # deterministic
    assert abs(float(np.linalg.norm(v1[0])) - 1.0) < 1e-5  # L2-normalized


def test_empty_text_gives_zero_vector():
    e = HashingEmbedder(dim=64)
    v = e.encode([""])
    assert float(np.linalg.norm(v[0])) == 0.0  # no divide-by-zero blow-up


def test_vector_index_ranks_relevant_document_first():
    idx = VectorIndex(HashingEmbedder(dim=512))
    idx.add(
        [
            Document("a", "bamboo desk organizer with five compartments"),
            Document("b", "black rollerball pen with a metal barrel"),
            Document("c", "a5 dotted notebook, fountain pen friendly"),
        ]
    )
    hits = idx.search("bamboo organizer for my desk", k=2)
    assert hits[0][0].id == "a"


def test_vector_index_filter_fn_restricts_results():
    idx = VectorIndex(HashingEmbedder(dim=256))
    idx.add(
        [
            Document("a", "pen one", {"cat": "pens"}),
            Document("b", "notebook one", {"cat": "notebooks"}),
        ]
    )
    hits = idx.search("item", k=10, filter_fn=lambda d: d.metadata["cat"] == "pens")
    assert [d.id for d, _ in hits] == ["a"]


def test_index_save_and_load_roundtrip(tmp_path):
    e = HashingEmbedder(dim=128)
    idx = VectorIndex(e)
    idx.add([Document("a", "hello world"), Document("b", "foo bar")])
    idx.save(tmp_path)

    loaded = VectorIndex.load(tmp_path, e)
    assert [d.id for d in loaded.documents] == ["a", "b"]
    # search still works after reload
    assert loaded.search("hello", k=1)[0][0].id == "a"


def test_split_markdown_sections_drops_preamble():
    text = "# Title\n<!-- comment -->\n### Q1\nbody one\n### Q2\nbody two\n"
    sections = split_markdown_sections(text, "### ")
    assert [t for t, _ in sections] == ["Q1", "Q2"]
    assert sections[0][1] == "body one"
