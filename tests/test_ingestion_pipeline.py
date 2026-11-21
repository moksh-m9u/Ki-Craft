"""Tests for src/ingestion/pipeline.py: chunk exports become stored records."""

import base64
import json
from pathlib import Path

import langsmith
import pytest

pytest.importorskip("langchain_chroma")

from src.ingestion import pipeline
from src.ingestion.pipeline import documents_from_export, run_complete_ingestion_pipeline

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


def test_records_have_flat_metadata_and_figure_files(tmp_path):
    images_dir = tmp_path / "images"
    docs = documents_from_export(_export(), "ne555", images_dir)

    assert [d.page_content for d in docs] == ["summary one", "summary two"]
    assert set(docs[1].metadata) == {
        "source", "chunk_id", "raw_text", "tables_html", "image_paths", "has_table", "has_image",
    }
    assert docs[0].metadata["image_paths"] == "[]"
    assert not docs[0].metadata["has_image"] and not docs[0].metadata["has_table"]

    paths = json.loads(docs[1].metadata["image_paths"])
    assert paths == [str(images_dir / "ne555_c2_img0.jpg"), str(images_dir / "ne555_c2_img1.jpg")]
    assert (images_dir / "ne555_c2_img0.jpg").read_bytes() == JPEG
    assert json.loads(docs[1].metadata["tables_html"]) == ["<table/>"]
    assert docs[1].metadata["source"] == "ne555" and docs[1].metadata["chunk_id"] == 2


def test_stored_paths_keep_the_images_dir_as_given(tmp_path, monkeypatch):
    """A relative images folder gives repo-relative paths, like the shipped store."""
    monkeypatch.chdir(tmp_path)
    docs = documents_from_export(_export(), "ne555", "dbv2/images")
    assert json.loads(docs[1].metadata["image_paths"])[0] == "dbv2/images/ne555_c2_img0.jpg"


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

    run_complete_ingestion_pipeline(
        "data/ne555.pdf",
        persist_directory=str(tmp_path / "store"),
        images_dir=tmp_path / "img",
        export_dir=tmp_path / "json",
    )
    assert (tmp_path / "json" / "ne555.json").exists()
    assert seen["document"] == "ne555: header"
    records = _records(tmp_path)
    assert records["sources"] == ["ne555", "ne555"]
    assert records["documents"] == ["summary one", "summary two"]
    assert (tmp_path / "img" / "ne555_c2_img0.jpg").read_bytes() == JPEG


class _Embedder:
    """An offline embedder for a temporary Chroma store; can be told to fail."""

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


def _store(tmp_path, monkeypatch, embedder: _Embedder | None = None) -> _Embedder:
    """Point store_export at a temporary Chroma store and an offline embedder."""
    embedder = embedder or _Embedder()
    monkeypatch.setattr(pipeline, "load_embedding_model", lambda: embedder)
    (tmp_path / "ne555.json").write_text(json.dumps(_export()))
    return embedder


def _records(tmp_path) -> dict:
    from langchain_chroma import Chroma

    db = Chroma(persist_directory=str(tmp_path / "store"), embedding_function=_Embedder())
    found = db.get(include=["metadatas", "documents", "embeddings"])
    return {"ids": found["ids"], "sources": sorted(m["source"] for m in found["metadatas"]),
            "documents": sorted(found["documents"]), "embeddings": found["embeddings"]}


def _put(tmp_path, monkeypatch, replace: bool = False, embedder: _Embedder | None = None):
    _store(tmp_path, monkeypatch, embedder)
    return pipeline.store_export(tmp_path / "ne555.json", str(tmp_path / "store"), tmp_path / "img", replace=replace)


def test_store_export_adds_its_records_without_summarising(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "summarise_chunks", lambda *a: pytest.fail("no summarising"))
    embedder = _Embedder()
    _put(tmp_path, monkeypatch, embedder=embedder)
    records = _records(tmp_path)
    assert records["sources"] == ["ne555", "ne555"]
    assert records["documents"] == ["summary one", "summary two"]
    assert [list(v) for v in records["embeddings"]] == [pytest.approx([11.0, 1.0, 0.0])] * 2
    assert embedder.calls == 1, "every summary is embedded once"
    assert (tmp_path / "img" / "ne555_c2_img0.jpg").read_bytes() == JPEG


def test_store_export_refuses_a_document_already_stored(tmp_path, monkeypatch):
    _put(tmp_path, monkeypatch)
    before = _records(tmp_path)["ids"]
    with pytest.raises(FileExistsError, match="--replace"):
        _put(tmp_path, monkeypatch)
    assert _records(tmp_path)["ids"] == before


def test_replace_swaps_the_records_and_removes_stale_figures(tmp_path, monkeypatch):
    _put(tmp_path, monkeypatch)
    old_ids = set(_records(tmp_path)["ids"])
    (tmp_path / "img" / "ne555_c9_img0.jpg").write_bytes(b"old figure of a chunk that no longer exists")
    (tmp_path / "img" / "lm555_c1_img0.jpg").write_bytes(b"another document")

    _put(tmp_path, monkeypatch, replace=True)
    records = _records(tmp_path)
    assert records["sources"] == ["ne555", "ne555"]
    assert not old_ids & set(records["ids"])
    assert not (tmp_path / "img" / "ne555_c9_img0.jpg").exists()
    assert (tmp_path / "img" / "ne555_c2_img0.jpg").read_bytes() == JPEG
    assert (tmp_path / "img" / "lm555_c1_img0.jpg").exists(), "other documents are untouched"


def test_failed_embedding_leaves_the_stored_document_as_it_was(tmp_path, monkeypatch):
    """Embedding runs before anything is deleted."""
    _put(tmp_path, monkeypatch)
    before = _records(tmp_path)["ids"]
    with pytest.raises(RuntimeError, match="embedding service down"):
        _put(tmp_path, monkeypatch, replace=True, embedder=_Embedder(fail=True))
    assert _records(tmp_path)["ids"] == before
    assert (tmp_path / "img" / "ne555_c2_img0.jpg").exists()


def test_store_export_writes_into_the_configured_models_collection(tmp_path, monkeypatch):
    """With OpenRouter configured, records go to that model's collection; Granite's is untouched."""
    import chromadb

    import src.embed
    import src.retrieval.search

    _put(tmp_path, monkeypatch)
    granite_ids = _records(tmp_path)["ids"]
    for module in (src.embed, src.retrieval.search):
        monkeypatch.setattr(module, "EMBEDDING_BACKEND", "openrouter")
    monkeypatch.setattr(src.embed, "OPENROUTER_EMBEDDING_MODEL", "vendor/model")

    _put(tmp_path, monkeypatch)

    client = chromadb.PersistentClient(path=str(tmp_path / "store"))
    assert sorted(c.name for c in client.list_collections()) == ["langchain", "vendor-model"]
    assert client.get_collection("vendor-model").count() == 2
    assert client.get_collection("vendor-model").metadata["embedding_model"] == "vendor/model"
    assert _records(tmp_path)["ids"] == granite_ids


@pytest.mark.parametrize(
    "broken, message",
    [
        (lambda e: e[1].update(chunk_id=5), "chunk ids"),
        (lambda e: e[0].update(enhanced_content=" "), "no summary"),
        (lambda e: e[1]["metadata"]["original_content"].update(images_base64=["not base64!"]), "does not decode"),
    ],
)
def test_a_broken_export_is_rejected_before_any_change(tmp_path, monkeypatch, broken, message):
    export = _export()
    broken(export)
    embedder = _store(tmp_path, monkeypatch)
    (tmp_path / "ne555.json").write_text(json.dumps(export))
    with pytest.raises(ValueError, match=message):
        pipeline.store_export(tmp_path / "ne555.json", str(tmp_path / "store"), tmp_path / "img")
    assert embedder.calls == 0
    assert not (tmp_path / "img").exists()
