"""Fixtures for the API tests: the app with every heavy dependency faked."""

from __future__ import annotations

import langsmith
import pytest
from fastapi.testclient import TestClient

import api.service
import src.db.connection
import src.documents
import src.embed
import src.llm
import src.retrieval.search
from api.main import app
from tests.api.fakes import JPEG_BYTES, FakeBackend, FakeStore, forbidden


@pytest.fixture
def backend(monkeypatch, tmp_path) -> FakeBackend:
    """Patch every heavy dependency with fakes and turn tracing off.

    Figure files live under ``tmp_path``, which stands in for the repository
    root, so no test reads the real ``dbv2/images``.
    """
    fake = FakeBackend()
    (tmp_path / "dbv2" / "images").mkdir(parents=True)
    (tmp_path / "dbv2" / "images" / "tlv1117_c1_img0.jpg").write_bytes(JPEG_BYTES)
    (tmp_path / "secret.txt").write_text("outside the images folder")
    monkeypatch.setattr(src.documents, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(api.service, "load_embedding_model", lambda: "fake-embeddings")
    store = FakeStore()
    monkeypatch.setattr(api.service, "create_db_engine", lambda url: "fake-engine")
    monkeypatch.setattr(api.service, "ping", fake.ping)
    monkeypatch.setattr(api.service, "PostgresRetriever", fake.retriever)
    monkeypatch.setattr(api.service, "list_documents", store.list_documents)
    monkeypatch.setattr(api.service, "document_chunks", store.document_chunks)
    monkeypatch.setattr(api.service, "figure_path", store.figure_path)
    monkeypatch.setattr(src.db.connection, "create_engine", forbidden("create_engine"))
    monkeypatch.setattr(src.llm, "ChatGoogleGenerativeAI", fake.make_llm)
    monkeypatch.setattr(src.llm, "ChatOpenAI", fake.make_llm)
    monkeypatch.setattr(src.embed, "WatsonxEmbeddings", forbidden("WatsonxEmbeddings"))

    # A process-wide switch: it also reaches the threads TestClient and
    # asyncio.to_thread run the app in, which a context manager would not.
    langsmith.configure(enabled=False)
    yield fake
    langsmith.configure(enabled=None)


@pytest.fixture
def client(backend: FakeBackend):
    """A TestClient whose ``with`` block runs the app's lifespan."""
    with TestClient(app) as test_client:
        yield test_client
