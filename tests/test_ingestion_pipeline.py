"""Tests for src/ingestion/pipeline.py: chunk exports become stored chunks.

The two functions that run SQL (``is_stored``, ``write_document``) are
replaced by an in-memory database, so no test needs Postgres; their SQL is
checked live (see AGENTS.md, Checkpoint 6)."""

import base64
import json
import uuid
from pathlib import Path

import langsmith
import pytest

from src.ingestion import pipeline
from src.ingestion.pipeline import rows_from_export, run_complete_ingestion_pipeline

JPEG = b"\xff\xd8\xff\xe0fig"


@pytest.fixture(autouse=True)
def _no_tracing():
    """The pipeline functions are traceable; keep them off the network."""
    langsmith.configure(enabled=False)
    yield
    langsmith.configure(enabled=None)


def _export() -> list[dict]:
    """A two-chunk export as export_chunks_to_json writes it."""
    return [
        {
            "chunk_id": 1,
            "enhanced_content": "summary one",
            "metadata": {"original_content": {"raw_text": "text one", "tables_html": [], "images_base64": []}},
        },
        {
            "chunk_id": 2,
            "enhanced_content": "summary two",
            "metadata": {
                "original_content": {
                    "raw_text": "text two",
                    "tables_html": ["<table/>"],
                    "images_base64": [base64.b64encode(JPEG).decode(), base64.b64encode(b"second").decode()],
                }
            },
        },
    ]


def test_rows_carry_the_export_and_figure_files(tmp_path):
    images_dir = tmp_path / "images"
    chunk_rows, image_rows = rows_from_export(_export(), "ne555", [[1.0], [2.0]], images_dir)

    assert [r["summary"] for r in chunk_rows] == ["summary one", "summary two"]
    assert [r["chunk_id"] for r in chunk_rows] == [1, 2]
    assert [r["embedding"] for r in chunk_rows] == [[1.0], [2.0]]
    assert chunk_rows[1]["tables_html"] == ["<table/>"] and chunk_rows[0]["tables_html"] == []
    assert chunk_rows[1]["raw_text"] == "text two"
    assert all(isinstance(r["id"], uuid.UUID) for r in chunk_rows)
    assert image_rows == [
        {"chunk_id": chunk_rows[1]["id"], "position": 0, "path": str(images_dir / "ne555_c2_img0.jpg")},
        {"chunk_id": chunk_rows[1]["id"], "position": 1, "path": str(images_dir / "ne555_c2_img1.jpg")},
    ]
    assert (images_dir / "ne555_c2_img0.jpg").read_bytes() == JPEG


def test_stored_paths_keep_the_images_dir_as_given(tmp_path, monkeypatch):
    """A relative images folder gives repo-relative paths, like the shipped figures."""
    monkeypatch.chdir(tmp_path)
    _, image_rows = rows_from_export(_export(), "ne555", [[1.0], [2.0]], "dbv2/images")
    assert image_rows[0]["path"] == "dbv2/images/ne555_c2_img0.jpg"


def test_existing_export_stops_before_any_work(tmp_path, monkeypatch):
    (tmp_path / "ne555.json").write_text("[]")

    def forbidden(*args, **kwargs):
        raise AssertionError("no step should run")

    monkeypatch.setattr(pipeline, "partition_document", forbidden)
    with pytest.raises(FileExistsError):
        run_complete_ingestion_pipeline("data/ne555.pdf", export_dir=tmp_path)


def test_pipeline_writes_the_export_then_stores_it(tmp_path, monkeypatch):
    """A PDF is stored through store_export, the same path as an existing export."""
    seen = {}
    monkeypatch.setattr(pipeline, "partition_document", lambda path: ["element"])
    monkeypatch.setattr(pipeline, "create_chunks_by_title", lambda elements: ["chunk"])
    monkeypatch.setattr(pipeline, "document_context", lambda elements, source: f"{source}: header")
    monkeypatch.setattr(pipeline, "remove_page_decoration", lambda elements: elements)
    monkeypatch.setattr(
        pipeline, "summarise_chunks",
        lambda chunks, backend, document: seen.update(document=document) or ["summarised"],
    )
    monkeypatch.setattr(
        pipeline, "export_chunks_to_json",
        lambda docs, filename: Path(filename).write_text(json.dumps(_export())),
    )
    monkeypatch.setattr(pipeline, "load_embedding_model", _Embedder)
    database = _Database(monkeypatch)

    run_complete_ingestion_pipeline(
        "data/ne555.pdf",
        images_dir=tmp_path / "img",
        export_dir=tmp_path / "json",
        engine="fake-engine",
    )
    assert (tmp_path / "json" / "ne555.json").exists()
    assert seen["document"] == "ne555: header"
    assert [r["summary"] for r in database.documents["ne555"]["chunks"]] == ["summary one", "summary two"]
    assert (tmp_path / "img" / "ne555_c2_img0.jpg").read_bytes() == JPEG


class _Embedder:
    """An offline embedder; can be told to fail."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding service down")
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


class _Database:
    """An in-memory stand-in for ``is_stored`` and ``write_document``."""

    def __init__(self, monkeypatch):
        self.documents: dict[str, dict] = {}
        monkeypatch.setattr(pipeline, "is_stored", lambda engine, source: source in self.documents)
        monkeypatch.setattr(pipeline, "write_document", self.write)

    def write(self, engine, source, chunk_rows, image_rows):
        """Replace the document, as one transaction would."""
        self.documents[source] = {"chunks": chunk_rows, "images": image_rows}

    def ids(self, source: str = "ne555") -> list:
        """The stored record ids of one document."""
        return [r["id"] for r in self.documents.get(source, {}).get("chunks", [])]


def _store(tmp_path, monkeypatch, embedder: _Embedder | None = None) -> _Embedder:
    """Point store_export at an offline embedder and write the export file."""
    embedder = embedder or _Embedder()
    monkeypatch.setattr(pipeline, "load_embedding_model", lambda: embedder)
    (tmp_path / "ne555.json").write_text(json.dumps(_export()))
    return embedder


@pytest.fixture
def database(monkeypatch) -> _Database:
    """An empty in-memory database."""
    return _Database(monkeypatch)


def _put(tmp_path, monkeypatch, replace: bool = False, embedder: _Embedder | None = None):
    _store(tmp_path, monkeypatch, embedder)
    return pipeline.store_export(tmp_path / "ne555.json", tmp_path / "img", replace=replace, engine="fake-engine")


def test_store_export_adds_its_chunks_without_summarising(tmp_path, monkeypatch, database):
    monkeypatch.setattr(pipeline, "summarise_chunks", lambda *a: pytest.fail("no summarising"))
    embedder = _Embedder()
    assert _put(tmp_path, monkeypatch, embedder=embedder) == 2
    stored = database.documents["ne555"]
    assert [r["summary"] for r in stored["chunks"]] == ["summary one", "summary two"]
    assert [r["embedding"] for r in stored["chunks"]] == [[11.0, 1.0, 0.0], [11.0, 1.0, 0.0]]
    assert len(stored["images"]) == 2
    assert embedder.calls == 1, "every summary is embedded once"
    assert (tmp_path / "img" / "ne555_c2_img0.jpg").read_bytes() == JPEG


def test_store_export_refuses_a_document_already_stored(tmp_path, monkeypatch, database):
    _put(tmp_path, monkeypatch)
    before = database.ids()
    embedder = _Embedder()
    with pytest.raises(FileExistsError, match="--replace"):
        _put(tmp_path, monkeypatch, embedder=embedder)
    assert database.ids() == before
    assert embedder.calls == 0, "refused before the embedding call"


def test_replace_swaps_the_chunks_and_removes_stale_figures(tmp_path, monkeypatch, database):
    _put(tmp_path, monkeypatch)
    old_ids = set(database.ids())
    (tmp_path / "img" / "ne555_c9_img0.jpg").write_bytes(b"old figure of a chunk that no longer exists")
    (tmp_path / "img" / "lm555_c1_img0.jpg").write_bytes(b"another document")

    _put(tmp_path, monkeypatch, replace=True)
    assert len(database.ids()) == 2
    assert not old_ids & set(database.ids()), "replaced chunks get new record ids"
    assert not (tmp_path / "img" / "ne555_c9_img0.jpg").exists()
    assert (tmp_path / "img" / "ne555_c2_img0.jpg").read_bytes() == JPEG
    assert (tmp_path / "img" / "lm555_c1_img0.jpg").exists(), "other documents are untouched"


def test_failed_embedding_leaves_the_stored_document_as_it_was(tmp_path, monkeypatch, database):
    """Embedding runs before anything is written or deleted."""
    _put(tmp_path, monkeypatch)
    before = database.ids()
    (tmp_path / "img" / "ne555_c9_img0.jpg").write_bytes(b"old figure")
    with pytest.raises(RuntimeError, match="embedding service down"):
        _put(tmp_path, monkeypatch, replace=True, embedder=_Embedder(fail=True))
    assert database.ids() == before
    assert (tmp_path / "img" / "ne555_c9_img0.jpg").exists()


def test_failed_database_write_keeps_old_figure_files(tmp_path, monkeypatch, database):
    """Stale figure files are removed only after the transaction succeeds."""
    _put(tmp_path, monkeypatch)
    (tmp_path / "img" / "ne555_c9_img0.jpg").write_bytes(b"old figure")

    def failing_write(*args):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(pipeline, "write_document", failing_write)
    with pytest.raises(RuntimeError, match="connection lost"):
        _put(tmp_path, monkeypatch, replace=True)
    assert (tmp_path / "img" / "ne555_c9_img0.jpg").exists()


@pytest.mark.parametrize(
    "broken, message",
    [
        (lambda e: e[1].update(chunk_id=5), "chunk ids"),
        (lambda e: e[0].update(enhanced_content=" "), "no summary"),
        (lambda e: e[1]["metadata"]["original_content"].update(images_base64=["not base64!"]), "does not decode"),
    ],
)
def test_a_broken_export_is_rejected_before_any_change(tmp_path, monkeypatch, database, broken, message):
    export = _export()
    broken(export)
    embedder = _store(tmp_path, monkeypatch)
    (tmp_path / "ne555.json").write_text(json.dumps(export))
    with pytest.raises(ValueError, match=message):
        pipeline.store_export(tmp_path / "ne555.json", tmp_path / "img", engine="fake-engine")
    assert embedder.calls == 0
    assert not (tmp_path / "img").exists()
    assert database.documents == {}
