"""Browse the stored corpus: which documents exist, their chunks in order, one figure's file.

Sits beside retrieval, not inside it: these reads use no embedding and no
vector search, only plain SQL on the Postgres tables (``src/db/schema.py``).
``api/service.py`` calls them for ``GET /v1/documents``,
``GET /v1/documents/{document_id}/chunks`` and
``GET /v1/images/{record_id}/{position}``; the dashboard's Chunk Explorer,
Compare and Analytics pages read the first two.

A document is identified by its ``source``, the PDF's file name without
``.pdf`` (``lm317``). Chunks come back in reading order, by the ``chunk_id``
stored at ingestion, which counts from 1 within each document. The database
counts, sorts and pages; Python only shapes the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from sqlalchemy import Engine, func, select

from src.db.schema import chunks, documents, images
from src.documents import document_from_row, stored_chunks_query


@dataclass(frozen=True)
class DocumentSummary:
    """One stored document and how much it holds.

    Attributes:
        id: The document's ``source`` name, such as ``lm317``.
        chunk_count: Stored chunks.
        image_count: Figures across those chunks.
        table_count: Tables across those chunks.
    """

    id: str
    chunk_count: int
    image_count: int
    table_count: int


def list_documents(engine: Engine) -> list[DocumentSummary]:
    """Summarise every stored document, sorted by name.

    One query: per document, count its chunks, add up the tables in each
    chunk's ``tables_html`` array, and count its figures. Figures are counted
    in a subquery per chunk, because joining ``images`` directly would repeat
    each chunk once per figure and inflate the other counts.

    Args:
        engine: The Postgres engine.

    Returns:
        One summary per document, including a document with no chunks.
    """
    figures_per_chunk = (
        select(func.count()).where(images.c.chunk_id == chunks.c.id).scalar_subquery()
    )
    query = (
        select(
            documents.c.source,
            func.count(chunks.c.id),
            func.coalesce(func.sum(figures_per_chunk), 0),
            func.coalesce(func.sum(func.cardinality(chunks.c.tables_html)), 0),
        )
        .select_from(documents)
        .outerjoin(chunks, chunks.c.document_id == documents.c.id)
        .group_by(documents.c.source)
        .order_by(documents.c.source)
    )
    with engine.connect() as connection:
        return [DocumentSummary(source, int(n), int(f), int(t)) for source, n, f, t in connection.execute(query)]


def document_chunks(engine: Engine, document_id: str, offset: int, limit: int) -> tuple[list[Document], int]:
    """Return one page of a document's chunks, in reading order.

    Args:
        engine: The Postgres engine.
        document_id: The document's ``source`` name.
        offset: Chunks to skip, counted from the start of the document.
        limit: Most chunks to return.

    Returns:
        ``(chunks, total)``: the page as ``Document`` objects (record id,
        summary as ``page_content``, original content in ``metadata``), and how
        many chunks the document holds. ``total`` is 0 for an unknown document.
    """
    in_document = documents.c.source == document_id
    count = select(func.count()).select_from(chunks.join(documents)).where(in_document)
    page = stored_chunks_query().where(in_document).order_by(chunks.c.chunk_id).offset(offset).limit(limit)
    with engine.connect() as connection:
        total = connection.execute(count).scalar_one()
        rows = list(connection.execute(page)) if total else []
    return [document_from_row(row) for row in rows], total


def figure_path(engine: Engine, record_id: str, position: int) -> str | None:
    """The stored path of one figure, or None when the chunk or position does not exist.

    Args:
        engine: The Postgres engine.
        record_id: The chunk's record id (a UUID string).
        position: 0-based position of the figure within the chunk.

    Returns:
        The path as stored at ingestion, relative to the repository root. The
        caller still checks it lies inside ``IMAGES_DIR``
        (``src/documents.py:figure_file``).
    """
    query = select(images.c.path).where(images.c.chunk_id == record_id, images.c.position == position)
    with engine.connect() as connection:
        return connection.execute(query).scalar_one_or_none()
