"""Tests for copying a collection into another model's collection (``src/ingestion/reembed.py``).

Each test builds a temporary Chroma store with a small Granite-like
``langchain`` collection and embeds with an offline fake, so nothing touches
``dbv2`` or the network.
"""

from __future__ import annotations

import json
import sys

import chromadb
import pytest

import scripts.reembed as cli
import src.embed as embed
import src.retrieval.search as search
from src.ingestion.reembed import describe_collections, reembed_collection

_RECORDS = 5


class _Embedder:
    """Gives 3-number vectors; can fail on a given call, and records what it embedded."""

    def __init__(self, fail_on_call: int | None = None):
        self.fail_on_call = fail_on_call
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(self.batches) + 1 == self.fail_on_call:
            raise RuntimeError("429 rate limit")
        self.batches.append(texts)
        return [[float(len(text)), 1.0, 0.0] for text in texts]


@pytest.fixture
def store(tmp_path, monkeypatch) -> str:
    """A store holding the original collection, with OpenRouter configured."""
    for module in (embed, search):
        monkeypatch.setattr(module, "EMBEDDING_BACKEND", "openrouter")
    monkeypatch.setattr(embed, "OPENROUTER_EMBEDDING_MODEL", "vendor/model:free")
    path = str(tmp_path / "store")
    original = chromadb.PersistentClient(path=path).create_collection("langchain", metadata={"hnsw:space": "cosine"})
    original.add(
        ids=[f"id-{n}" for n in range(_RECORDS)],
        embeddings=[[0.1] * 4 for _ in range(_RECORDS)],
        documents=[f"summary {n}" for n in range(_RECORDS)],
        metadatas=[
            {"source": "ne555", "chunk_id": n + 1, "image_paths": json.dumps([f"dbv2/images/ne555_c{n + 1}_img0.jpg"])}
            for n in range(_RECORDS)
        ],
    )
    return path


def _collections(path: str) -> dict:
    client = chromadb.PersistentClient(path=path)
    return {c.name: c for c in client.list_collections()}


def test_copies_every_record_under_the_same_id_with_the_new_vectors(store):
    report = reembed_collection(_Embedder(), store, batch_size=2)

    assert report["status"] == "created"
    assert (report["target"], report["records"], report["embedded"], report["batches"]) == ("vendor-model", 5, 5, 3)
    assert (report["figures"], report["dimension"]) == (5, 3)
    collections = _collections(store)
    assert sorted(collections) == ["langchain", "vendor-model"]
    copy = collections["vendor-model"].get(include=["documents", "metadatas", "embeddings"])
    original = collections["langchain"].get(include=["documents", "metadatas", "embeddings"])
    assert sorted(copy["ids"]) == sorted(original["ids"])
    by_id = {i: (d, m) for i, d, m in zip(original["ids"], original["documents"], original["metadatas"])}
    assert {i: (d, m) for i, d, m in zip(copy["ids"], copy["documents"], copy["metadatas"])} == by_id
    assert all(len(v) == 3 for v in copy["embeddings"])
    assert all(len(v) == 4 for v in original["embeddings"]), "the source keeps its own vectors"
    assert collections["vendor-model"].metadata["embedding_model"] == "vendor/model"


def test_an_interrupted_run_resumes_without_embedding_finished_batches_again(store):
    with pytest.raises(RuntimeError, match="429"):
        reembed_collection(_Embedder(fail_on_call=2), store, batch_size=2)
    collections = _collections(store)
    assert sorted(collections) == ["langchain", "vendor-model-partial"], "the API never sees a half-built copy"
    assert collections["vendor-model-partial"].count() == 2

    retry = _Embedder()
    report = reembed_collection(retry, store, batch_size=2)

    assert report["status"] == "created"
    assert report["embedded"] == 3
    assert sum(len(batch) for batch in retry.batches) == 3
    assert sorted(_collections(store)) == ["langchain", "vendor-model"]
    assert _collections(store)["vendor-model"].count() == 5


def test_a_finished_collection_is_checked_not_rebuilt(store):
    reembed_collection(_Embedder(), store)
    again = _Embedder()

    report = reembed_collection(again, store)

    assert report["status"] == "exists"
    assert again.batches == []


def test_a_finished_collection_that_no_longer_matches_is_reported(store):
    reembed_collection(_Embedder(), store)
    _collections(store)["langchain"].add(ids=["id-new"], embeddings=[[0.1] * 4], documents=["new"], metadatas=[{"a": 1}])

    with pytest.raises(RuntimeError, match="no longer matches.*delete it"):
        reembed_collection(_Embedder(), store)


def test_dry_run_embeds_and_writes_nothing(store):
    report = reembed_collection(None, store, batch_size=2, dry_run=True)

    assert report["status"] == "dry-run"
    assert (report["records"], report["batches"], report["estimated_tokens"]) == (5, 3, 45 // 4)
    assert sorted(_collections(store)) == ["langchain"]


def test_the_configured_collection_cannot_be_its_own_source(store, monkeypatch):
    monkeypatch.setattr(embed, "EMBEDDING_BACKEND", "watsonx")

    with pytest.raises(ValueError, match="already is the collection"):
        reembed_collection(_Embedder(), store)


def test_status_lists_each_collection_with_its_model(store):
    reembed_collection(_Embedder(), store)

    rows = describe_collections(store)

    assert rows == [
        {"name": "langchain", "embedding_model": None, "records": 5, "dimension": 4, "configured": False},
        {"name": "vendor-model", "embedding_model": "vendor/model", "records": 5, "dimension": 3, "configured": True},
    ]


def _cli(monkeypatch, *args: str) -> int:
    """Run ``python -m scripts.reembed`` with ``args``; return its exit code."""
    monkeypatch.setattr(sys, "argv", ["reembed", *args])
    return cli.main()


def test_cli_status_lists_the_collections_without_building_an_embedder(store, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_embedding_model", lambda: pytest.fail("--status makes no network call"))

    assert _cli(monkeypatch, "--status", "--store", store) == 0
    out = capsys.readouterr().out
    assert "langchain" in out and "granite (unrecorded)" in out
    assert "The configured collection 'vendor-model' does not exist yet: run python -m scripts.reembed" in out


def test_cli_dry_run_reports_and_writes_nothing(store, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_embedding_model", lambda: pytest.fail("--dry-run makes no network call"))

    assert _cli(monkeypatch, "--dry-run", "--store", store, "--batch-size", "2") == 0
    out = capsys.readouterr().out
    assert "dry-run: langchain -> vendor-model (vendor/model:free)" in out
    assert "records 5, embedded now 0, batches 3" in out
    assert sorted(_collections(store)) == ["langchain"]


def test_cli_run_builds_the_collection_and_prints_the_report(store, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_embedding_model", _Embedder)

    assert _cli(monkeypatch, "--store", store, "--batch-size", "2") == 0
    out = capsys.readouterr().out
    assert "created: langchain -> vendor-model (vendor/model:free)" in out
    assert "figure references kept 5, vector dimension 3" in out


def test_cli_failure_exits_1_and_says_the_same_command_resumes(store, monkeypatch, caplog):
    monkeypatch.setattr(cli, "load_embedding_model", lambda: _Embedder(fail_on_call=1))

    assert _cli(monkeypatch, "--store", store) == 1
    assert "Re-embedding failed: RuntimeError: 429 rate limit" in caplog.text
    assert "run the same command again to resume" in caplog.text
