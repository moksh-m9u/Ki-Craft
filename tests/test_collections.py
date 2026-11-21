"""Tests for one collection per embedding model (``src/retrieval/search.py:load_vector_store``).

Each test builds a temporary Chroma store, so the real ``dbv2`` is never opened.
"""

from __future__ import annotations

import functools

import chromadb
import langsmith
import pytest
from fastapi.testclient import TestClient

import api.service
import src.embed as embed
import src.retrieval.search as search
from api.main import app
from src.retrieval.search import CollectionError, check_collection, load_vector_store


class _Embedder:
    """An offline embedder giving 2-number vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.fixture
def openrouter(monkeypatch):
    """Configure the OpenRouter backend with a known model."""
    for module in (embed, search):
        monkeypatch.setattr(module, "EMBEDDING_BACKEND", "openrouter")
    monkeypatch.setattr(embed, "OPENROUTER_EMBEDDING_MODEL", "vendor/model:free")


def _collection(path, name: str, metadata: dict, records: int = 1):
    """Create a collection directly with Chroma, holding ``records`` records."""
    collection = chromadb.PersistentClient(path=str(path)).create_collection(name, metadata=metadata)
    if records:
        collection.add(ids=[str(i) for i in range(records)], embeddings=[[1.0, 0.0]] * records,
                       documents=["summary"] * records)
    return collection


def _names(path) -> list[str]:
    return sorted(c.name for c in chromadb.PersistentClient(path=str(path)).list_collections())


def test_the_original_granite_collection_loads_for_granite(tmp_path):
    _collection(tmp_path, "langchain", {"hnsw:space": "cosine"})

    db = load_vector_store(_Embedder(), str(tmp_path))

    assert db._collection.name == "langchain"
    assert db.similarity_search("q", k=1)[0].page_content == "summary"


def test_openrouter_loads_its_own_collection_not_granites(tmp_path, openrouter):
    _collection(tmp_path, "langchain", {"hnsw:space": "cosine"})
    _collection(tmp_path, "vendor-model", {"hnsw:space": "cosine", "embedding_model": "vendor/model"}, records=2)

    db = load_vector_store(_Embedder(), str(tmp_path))

    assert db._collection.name == "vendor-model"
    assert db._collection.count() == 2


def test_a_missing_collection_is_an_error_and_is_not_created(tmp_path, openrouter):
    _collection(tmp_path, "langchain", {"hnsw:space": "cosine"})

    with pytest.raises(CollectionError, match=r"no collection for embedding model vendor/model:free.*scripts\.reembed"):
        load_vector_store(_Embedder(), str(tmp_path))
    assert _names(tmp_path) == ["langchain"]


def test_an_empty_collection_is_an_error(tmp_path, openrouter):
    _collection(tmp_path, "vendor-model", {"hnsw:space": "cosine", "embedding_model": "vendor/model"}, records=0)

    with pytest.raises(CollectionError, match=r"holds no records\. Delete it, then rebuild it with: python -m scripts\.reembed"):
        load_vector_store(_Embedder(), str(tmp_path))


def test_a_collection_made_by_another_model_is_refused(tmp_path, openrouter):
    _collection(tmp_path, "vendor-model", {"hnsw:space": "cosine", "embedding_model": "other/model"})

    with pytest.raises(CollectionError, match="made by other/model, but the configured model is vendor/model:free"):
        load_vector_store(_Embedder(), str(tmp_path))


def test_a_collection_without_a_recorded_model_counts_as_granite_only_under_its_name():
    check_collection("langchain", {"hnsw:space": "cosine"}, embed.EMBEDDING_MODEL)
    with pytest.raises(CollectionError, match="made by ibm/granite"):
        check_collection("langchain", {"hnsw:space": "cosine"}, "vendor/model")
    with pytest.raises(CollectionError, match="an unrecorded model"):
        check_collection("vendor-model", {"hnsw:space": "cosine"}, "vendor/model")


def test_ingestion_creates_the_collection_with_its_model(tmp_path, openrouter):
    db = load_vector_store(_Embedder(), str(tmp_path), create=True)

    assert db._collection.name == "vendor-model"
    assert db._collection.metadata == {
        "hnsw:space": "cosine",
        "embedding_model": "vendor/model",
        "embedding_backend": "openrouter",
    }


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """The real API, opening collections in a temporary store with an offline embedder."""
    monkeypatch.setattr(api.service, "load_embedding_model", _Embedder)
    monkeypatch.setattr(api.service, "load_vector_store", functools.partial(load_vector_store, persist_directory=str(tmp_path)))
    langsmith.configure(enabled=False)
    with TestClient(app) as client:
        yield client
    langsmith.configure(enabled=None)


def test_api_health_is_503_and_names_the_fix_when_the_models_collection_is_missing(tmp_path, openrouter, api_client):
    _collection(tmp_path, "langchain", {"hnsw:space": "cosine"})

    response = api_client.get("/health")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "no collection for embedding model vendor/model:free" in detail
    assert "python -m scripts.reembed" in detail
    assert _names(tmp_path) == ["langchain"], "serving never creates a collection"


def test_api_retrieves_from_the_configured_models_collection(tmp_path, openrouter, api_client):
    _collection(tmp_path, "langchain", {"hnsw:space": "cosine"}, records=3)
    _collection(tmp_path, "vendor-model", {"hnsw:space": "cosine", "embedding_model": "vendor/model"}, records=2)

    response = api_client.post("/v1/retrieve", json={"query": "q", "top_k": 5})

    assert response.status_code == 200
    assert response.json()["num_chunks"] == 2, "the 2 records of vendor-model, not Granite's 3"
