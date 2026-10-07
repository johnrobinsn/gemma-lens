"""Integration test: boot the FastAPI app, hit /search, verify shape."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gemma_lens import config
from gemma_lens.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not config.INDEX_PATH.exists():
        pytest.skip("No index.npz — run `uv run python -m gemma_lens.ingest` first")
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_root_serves_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Gemma-Lens" in r.text


def test_examples_endpoint(client: TestClient) -> None:
    r = client.get("/examples")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all({"label", "modality"} <= set(ex.keys()) for ex in data)


def test_text_search(client: TestClient) -> None:
    r = client.post("/search", json={"modality": "text", "text": "red cars", "k": 8})
    assert r.status_code == 200
    data = r.json()
    assert data["embed_ms"] >= 0
    assert data["query_ms"] >= data["embed_ms"]
    assert len(data["results"]) == 8
    for hit in data["results"]:
        assert hit["score"] >= -1 and hit["score"] <= 1
        assert hit["modality"] in ("image", "audio")


def test_bad_request_rejected(client: TestClient) -> None:
    r = client.post("/search", json={"modality": "text"})  # missing `text`
    assert r.status_code == 400
