"""Turn chunk exports into stored records, and run the PDF-to-Chroma pipeline.

A chunk export (``json/<source>.json``) holds each chunk's summary and its
original text, tables and figures (base64). :func:`documents_from_export`
writes each figure to a file under the images folder and returns one
LangChain ``Document`` per chunk with flat metadata: ``source``, ``chunk_id``,
``raw_text``, ``tables_html``, ``image_paths``, ``has_table``, ``has_image``.

:func:`store_export` is the one way records reach the store: it checks an
export, embeds its summaries and writes the records and figure files into
the configured embedding model's collection (``src/embed.py:collection_name``).
:func:`run_complete_ingestion_pipeline` turns one PDF into an export and then
hands it to :func:`store_export`, so a PDF and an existing export are stored
the same way.
"""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langsmith import traceable

from config.settings import (
    CHROMA_PERSIST_DIR,
    ENHANCEMENT_BACKEND,
    IMAGES_DIR,
    JSON_EXPORT_DIR,
)
from src.embed import load_embedding_model
from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import summarise_chunks
from src.ingestion.export import export_chunks_to_json
from src.ingestion.extract import document_context, partition_document, remove_page_decoration
from src.logger import get_logger
from src.retrieval.search import load_vector_store

logger = get_logger(__name__)


def documents_from_export(
    export: list[dict],
    source: str,
    images_dir: str | Path = IMAGES_DIR,
) -> list[Document]:
    """Write an export's figures to files and build one Document per chunk.

    Figure ``n`` of chunk ``c`` is written to ``<images_dir>/<source>_c<c>_img<n>.jpg``
    and its path, as ``images_dir`` was given, is stored in ``image_paths``.

    Args:
        export: Entries with ``chunk_id``, ``enhanced_content`` and
            ``metadata.original_content`` (``raw_text``, ``tables_html``,
            ``images_base64``), as :func:`export_chunks_to_json` writes them.
        source: Document name stored on every record and used in file names.
        images_dir: Folder the figure files are written to.

    Returns:
        Documents with the summary as ``page_content`` and flat metadata.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    for chunk in export:
        original = chunk["metadata"]["original_content"]
        chunk_id = chunk["chunk_id"]
        tables = original.get("tables_html", [])
        image_paths = []
        for idx, b64 in enumerate(original.get("images_base64", [])):
            path = images_dir / f"{source}_c{chunk_id}_img{idx}.jpg"
            path.write_bytes(base64.b64decode(b64))
            image_paths.append(str(path))
        documents.append(
            Document(
                page_content=chunk["enhanced_content"],
                metadata={
                    "source": source,
                    "chunk_id": chunk_id,
                    "raw_text": original.get("raw_text", ""),
                    "tables_html": json.dumps(tables, ensure_ascii=False),
                    "image_paths": json.dumps(image_paths, ensure_ascii=False),
                    "has_table": bool(tables),
                    "has_image": bool(image_paths),
                },
            )
        )
    return documents


def _check_export(export: list, source: str) -> None:
    """Raise ValueError unless ``export`` is safe to store: ids in order, summaries, figures."""
    if not isinstance(export, list) or not export:
        raise ValueError(f"{source}: the export holds no chunks")
    if [chunk.get("chunk_id") for chunk in export] != list(range(1, len(export) + 1)):
        raise ValueError(f"{source}: chunk ids are not 1..{len(export)} in order")
    for chunk in export:
        if not str(chunk.get("enhanced_content") or "").strip():
            raise ValueError(f"{source}: chunk {chunk['chunk_id']} has no summary")
        for b64 in chunk["metadata"]["original_content"].get("images_base64", []):
            try:
                base64.b64decode(b64, validate=True)
            except binascii.Error as exc:
                raise ValueError(f"{source}: chunk {chunk['chunk_id']} has a figure that does not decode") from exc


@traceable(run_type="chain", name="StoreExport")
def store_export(
    export_path: str | Path,
    persist_directory: str = CHROMA_PERSIST_DIR,
    images_dir: str | Path = IMAGES_DIR,
    replace: bool = False,
) -> Chroma:
    """Add one chunk export to the store: write its figures, embed its summaries.

    The records go into the configured embedding model's collection. The
    document's name (``source``) is the export's file name without
    ``.json``. The steps run in an order that keeps the store whole if one
    fails: the export is checked, every summary is embedded (the network
    call), and only then are the document's old records and figure files
    removed (with ``replace``) and the new ones written.

    Args:
        export_path: A ``<source>.json`` export, as :func:`export_chunks_to_json` writes it.
        persist_directory: Folder of the Chroma store.
        images_dir: Folder the figure files are written to.
        replace: Replace the document's records and figures if the store
            already holds it.

    Returns:
        The Chroma store holding the new records.

    Raises:
        ValueError: If the export is empty, its chunk ids are out of order, a
            summary is empty or a figure does not decode; nothing is changed.
        FileExistsError: If the store already holds ``source`` and ``replace``
            is false; nothing is changed.
    """
    source = Path(export_path).stem
    export = json.loads(Path(export_path).read_text(encoding="utf-8"))
    _check_export(export, source)

    embedding = load_embedding_model()
    db = load_vector_store(embedding, persist_directory, create=True)
    old_ids = db.get(where={"source": source}, include=[])["ids"]
    if old_ids and not replace:
        raise FileExistsError(f"{persist_directory} already holds {source}; pass --replace to replace it")

    vectors = embedding.embed_documents([chunk["enhanced_content"] for chunk in export])

    if old_ids:
        db.delete(ids=old_ids)
        stale = list(Path(images_dir).glob(f"{source}_c*_img*.jpg"))
        for path in stale:
            path.unlink()
        logger.info("Removed %d records and %d figure files of %s", len(old_ids), len(stale), source)

    documents = documents_from_export(export, source, images_dir)
    # The collection's own add takes the vectors computed above; LangChain's
    # add_documents would embed every summary a second time.
    db._collection.add(
        ids=[str(uuid.uuid4()) for _ in documents],
        embeddings=vectors,
        documents=[document.page_content for document in documents],
        metadatas=[document.metadata for document in documents],
    )
    logger.info("Stored %d chunks from %s", len(documents), export_path)
    return db


@traceable(run_type="chain", name="FullIngestionPipeline")
def run_complete_ingestion_pipeline(
    pdf_path: str | Path,
    persist_directory: str = CHROMA_PERSIST_DIR,
    images_dir: str | Path = IMAGES_DIR,
    export_dir: str | Path = JSON_EXPORT_DIR,
    backend: str = ENHANCEMENT_BACKEND,
) -> Chroma:
    """Ingest one PDF: partition, drop page decoration, chunk, summarise, export, store.

    The document's name (``source``) is the PDF file name without its suffix.
    The export is written first and then stored by :func:`store_export`. If
    storing fails, the export stays on disk, so the slow summaries are not
    lost: store it with ``python -m scripts.ingest json/<source>.json``.

    Args:
        pdf_path: The PDF to ingest.
        persist_directory: Folder of the Chroma store.
        images_dir: Folder the figure files are written to.
        export_dir: Folder the chunk export ``<source>.json`` is written to.
        backend: Summary backend, ``huggingface`` or ``local``.

    Returns:
        The Chroma store holding the new records.

    Raises:
        FileExistsError: If ``<export_dir>/<source>.json`` already exists, which
            means the document was ingested before, or the store already
            holds ``source``.
        SummaryError: If a chunk's summary fails; nothing is written then.
    """
    source = Path(pdf_path).stem
    export_path = Path(export_dir) / f"{source}.json"
    if export_path.exists():
        raise FileExistsError(f"{export_path} exists; {source} looks ingested already")

    elements = partition_document(pdf_path)
    document = document_context(elements, source)
    chunks = create_chunks_by_title(remove_page_decoration(elements))
    summarised = summarise_chunks(chunks, backend, document)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_chunks_to_json(summarised, export_path)
    return store_export(export_path, persist_directory, images_dir)
