"""Tests for the Postgres search in ``src/retrieval/search.py`` (no database: rows are faked)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

import src.retrieval.search as search
from src.documents import document_from_row


def _row(n: int, embedding: list[float], distance: float, images: list[str] | None = None, tables=None):
    """A fake row of ``stored_chunks_query`` plus embedding and distance."""
    return SimpleNamespace(
        id=uuid.UUID(int=n),
        chunk_id=n,
        raw_text=f"text {n}",
        tables_html=tables or [],
        summary=f"summary {n}",
        source="lm317",
        image_paths=images,
        embedding=np.array(embedding, dtype=np.float32),
        distance=distance,
    )


@pytest.fixture
def pool(monkeypatch):
    """Four chunks, nearest first: 1 and 2 are near-duplicates, 3 and 4 point elsewhere."""
    rows = [
        _row(1, [1.0, 0.0, 0.0], 0.00),
        _row(2, [0.99, 0.01, 0.0], 0.01),
        _row(3, [0.6, 0.8, 0.0], 0.40),
        _row(4, [0.5, 0.0, 0.86], 0.50),
    ]
    calls = []

    def fake_nearest(engine, vector, n):
        calls.append(n)
        return rows[:n]

    monkeypatch.setattr(search, "nearest_chunks", fake_nearest)
    return SimpleNamespace(rows=rows, calls=calls)


def test_similarity_returns_the_k_nearest(pool):
    found = search.search_postgres(None, [1.0, 0.0, 0.0], "similarity", 2)
    assert [d.metadata["chunk_id"] for d in found] == [1, 2]
    assert pool.calls == [2]


def test_mmr_skips_the_near_duplicate_and_keeps_nearest_first_order(pool, monkeypatch):
    monkeypatch.setattr(search, "RETRIEVAL_FETCH_K", 4)
    monkeypatch.setattr(search, "RETRIEVAL_LAMBDA_MULT", 0.3)
    found = search.search_postgres(None, [1.0, 0.0, 0.0], "mmr", 2)
    assert pool.calls == [4]
    # MMR picks chunk 1, then a chunk unlike it; chunk 2 is nearly a copy of 1.
    ids = [d.metadata["chunk_id"] for d in found]
    assert 1 in ids and 2 not in ids
    assert ids == sorted(ids)


def test_mmr_pool_is_never_smaller_than_k(pool, monkeypatch):
    monkeypatch.setattr(search, "RETRIEVAL_FETCH_K", 1)
    search.search_postgres(None, [1.0, 0.0, 0.0], "mmr", 3)
    assert pool.calls == [3]


def test_score_threshold_drops_chunks_below_it(pool, monkeypatch):
    monkeypatch.setattr(search, "RETRIEVAL_SCORE_THRESHOLD", 0.55)
    found = search.search_postgres(None, [1.0, 0.0, 0.0], "similarity_score_threshold", 4)
    assert [d.metadata["chunk_id"] for d in found] == [1, 2, 3]


def test_unknown_search_type_is_rejected(pool):
    with pytest.raises(ValueError, match="Unknown search type 'hybrid'"):
        search.search_postgres(None, [1.0], "hybrid", 2)


def test_retriever_embeds_the_question_once(pool):
    class FakeEmbeddings:
        """Records the questions it embeds."""

        def __init__(self):
            self.queries = []

        def embed_query(self, text):
            self.queries.append(text)
            return [1.0, 0.0, 0.0]

    embeddings = FakeEmbeddings()
    retriever = search.PostgresRetriever(engine=None, embedding_model=embeddings, k=1, search_type="similarity")
    assert [d.metadata["chunk_id"] for d in retriever.invoke("dropout?")] == [1]
    assert embeddings.queries == ["dropout?"]


def test_row_becomes_the_document_shape_the_prompt_reads():
    row = _row(7, [0.0], 0.1, images=["dbv2/images/lm317_c7_img0.jpg"], tables=["<table></table>"])
    document = document_from_row(row)
    assert document.id == str(uuid.UUID(int=7))
    assert document.page_content == "summary 7"
    assert document.metadata == {
        "source": "lm317",
        "chunk_id": 7,
        "raw_text": "text 7",
        "tables_html": ["<table></table>"],
        "image_paths": ["dbv2/images/lm317_c7_img0.jpg"],
        "has_table": True,
        "has_image": True,
    }


def test_chunk_without_figures_has_empty_lists():
    document = document_from_row(_row(8, [0.0], 0.1))
    assert (document.metadata["image_paths"], document.metadata["has_image"]) == ([], False)
