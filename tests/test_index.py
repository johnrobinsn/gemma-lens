"""Smoke tests for the search index."""

from __future__ import annotations

import numpy as np
import pytest

from gemma_lens import config
from gemma_lens.index import SearchIndex


@pytest.fixture(scope="module")
def index() -> SearchIndex:
    if not config.INDEX_PATH.exists():
        pytest.skip("No index.npz — run `uv run python -m gemma_lens.ingest` first")
    return SearchIndex()


def test_index_loads(index: SearchIndex) -> None:
    assert index.image_embeddings.shape[0] > 0
    assert index.audio_embeddings.shape[0] > 0
    assert index.image_embeddings.shape[1] == index.audio_embeddings.shape[1]


def test_vectors_are_normalized(index: SearchIndex) -> None:
    for name, t in [("image", index.image_embeddings), ("audio", index.audio_embeddings)]:
        norms = t.float().norm(dim=1).cpu().numpy()
        assert np.allclose(norms, 1.0, atol=1e-2), f"{name} embeddings not L2-normalized"


def test_search_returns_k(index: SearchIndex) -> None:
    dim = index.image_embeddings.shape[1]
    q = np.random.randn(dim).astype(np.float32)
    q /= np.linalg.norm(q)
    hits = index.search(q, k=16)
    assert len(hits) == 16
    assert all({"id", "modality", "score", "filename", "url"} <= set(h.keys()) for h in hits)


def test_search_ranked(index: SearchIndex) -> None:
    dim = index.image_embeddings.shape[1]
    q = np.random.randn(dim).astype(np.float32)
    q /= np.linalg.norm(q)
    hits = index.search(q, k=16)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
