"""Turn chunk exports into stored chunks, and run the PDF-to-Postgres pipeline.

A chunk export (``json/<source>.json``) holds each chunk's summary and its
original text, tables and figures (base64). :func:`rows_from_export` writes
each figure to a file under the images folder and turns the export into rows
for the ``chunks`` and ``images`` tables (``src/db/schema.py``).

:func:`store_export` is the one way new chunks reach the database: it checks an
export, embeds its summaries, writes the figure files and stores the rows in
one transaction (``src/db/store.py:write_document``). :func:`run_complete_ingestion_pipeline` turns one PDF into an
export and then hands it to :func:`store_export`, so a PDF and an existing
export are stored the same way.

The figure bytes stay files on disk (``IMAGES_DIR``) and the database holds
their paths; object storage replaces the folder when uploads arrive
(Checkpoint 7).
"""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from pathlib import Path

from langsmith import traceable
from sqlalchemy import Engine

from config.settings import (
    DATABASE_URL_DIRECT,
    ENHANCEMENT_BACKEND,
    IMAGES_DIR,
    JSON_EXPORT_DIR,
)
from src.db.connection import create_db_engine
from src.db.store import is_stored, write_document
from src.embed import load_embedding_model
from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import summarise_chunks
from src.ingestion.export import export_chunks_to_json
from src.ingestion.extract import document_context, partition_document, remove_page_decoration
from src.logger import get_logger

logger = get_logger(__name__)


def rows_from_export(
    export: list[dict],
    source: str,
    vectors: list[list[float]],
    images_dir: str | Path = IMAGES_DIR,
) -> tuple[list[dict], list[dict]]:
    """Write an export's figures to files and build its ``chunks`` and ``images`` rows.

    Figure ``n`` of chunk ``c`` is written to ``<images_dir>/<source>_c<c>_img<n>.jpg``
    and its path, as ``images_dir`` was given, is stored in ``images.path``.
    Every chunk gets a new random record id; the caller adds ``document_id``.

    Args:
        export: Entries with ``chunk_id``, ``enhanced_content`` and
            ``metadata.original_content`` (``raw_text``, ``tables_html``,
            ``images_base64``), as :func:`export_chunks_to_json` writes them.
        source: Document name, used in the figure file names.
        vectors: One embedding per chunk, in export order.
        images_dir: Folder the figure files are written to.

    Returns:
        ``(chunk_rows, image_rows)``, ready to insert.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows: list[dict] = []
    image_rows: list[dict] = []
    for chunk, vector in zip(export, vectors, strict=True):
        original = chunk["metadata"]["original_content"]
        record_id = uuid.uuid4()
        for position, b64 in enumerate(original.get("images_base64", [])):
            path = images_dir / f"{source}_c{chunk['chunk_id']}_img{position}.jpg"
            path.write_bytes(base64.b64decode(b64))
            image_rows.append({"chunk_id": record_id, "position": position, "path": str(path)})
        chunk_rows.append(
            {
                "id": record_id,
                "chunk_id": chunk["chunk_id"],
                "raw_text": original.get("raw_text", ""),
                "tables_html": list(original.get("tables_html", [])),
                "summary": chunk["enhanced_content"],
                "embedding": vector,
            }
        )
    return chunk_rows, image_rows


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
    images_dir: str | Path = IMAGES_DIR,
    replace: bool = False,
    engine: Engine | None = None,
) -> int:
    """Add one chunk export to the database: embed its summaries, write its figures and rows.

    The document's name (``source``) is the export's file name without
    ``.json``. The steps run in an order that keeps the stored document
    whole if one fails: the export is checked and every summary is embedded
    (the network call) before anything is written. Then the figure files are
    written and the old document is replaced in one transaction. Figure files
    of the old version that the new one does not reuse are deleted last. One
    gap remains: if the transaction fails during ``--replace``, the old rows
    stay but figure files with the same name already hold the new images.

    Args:
        export_path: A ``<source>.json`` export, as :func:`export_chunks_to_json` writes it.
        images_dir: Folder the figure files are written to.
        replace: Replace the document if the database already holds it. Its
            chunks get new record ids, so its old figure URLs return 404.
        engine: The database to write to; ``DATABASE_URL_DIRECT`` when omitted.

    Returns:
        The number of chunks stored.

    Raises:
        ValueError: If the export is empty, its chunk ids are out of order, a
            summary is empty or a figure does not decode; nothing is changed.
        FileExistsError: If the database already holds ``source`` and
            ``replace`` is false; nothing is changed.
    """
    source = Path(export_path).stem
    export = json.loads(Path(export_path).read_text(encoding="utf-8"))
    _check_export(export, source)

    engine = engine or create_db_engine(DATABASE_URL_DIRECT, "DATABASE_URL_DIRECT")
    if is_stored(engine, source) and not replace:
        raise FileExistsError(f"the database already holds {source}; pass --replace to replace it")

    vectors = load_embedding_model().embed_documents([chunk["enhanced_content"] for chunk in export])

    old_files = set(Path(images_dir).glob(f"{source}_c*_img*.jpg"))
    chunk_rows, image_rows = rows_from_export(export, source, vectors, images_dir)
    write_document(engine, source, chunk_rows, image_rows)

    stale = old_files - {Path(row["path"]) for row in image_rows}
    for path in stale:
        path.unlink()
    logger.info("Stored %d chunks from %s (%d stale figure files removed)", len(chunk_rows), export_path, len(stale))
    return len(chunk_rows)


@traceable(run_type="chain", name="FullIngestionPipeline")
def run_complete_ingestion_pipeline(
    pdf_path: str | Path,
    images_dir: str | Path = IMAGES_DIR,
    export_dir: str | Path = JSON_EXPORT_DIR,
    backend: str = ENHANCEMENT_BACKEND,
    engine: Engine | None = None,
) -> int:
    """Ingest one PDF: partition, drop page decoration, chunk, summarise, export, store.

    The document's name (``source``) is the PDF file name without its suffix.
    The export is written first and then stored by :func:`store_export`. If
    storing fails, the export stays on disk, so the slow summaries are not
    lost: store it with ``python -m scripts.ingest json/<source>.json``.

    Args:
        pdf_path: The PDF to ingest.
        images_dir: Folder the figure files are written to.
        export_dir: Folder the chunk export ``<source>.json`` is written to.
        backend: Summary backend, ``huggingface`` or ``local``.
        engine: The database to write to; ``DATABASE_URL_DIRECT`` when omitted.

    Returns:
        The number of chunks stored.

    Raises:
        FileExistsError: If ``<export_dir>/<source>.json`` already exists, which
            means the document was ingested before, or the database already
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
    return store_export(export_path, images_dir, engine=engine)
