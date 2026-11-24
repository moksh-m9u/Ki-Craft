"""Read stored chunks out of Postgres as :class:`src.chunk.Chunk` objects.

The one SELECT every chunk read shares: retrieval (``src/retrieval/search.py``)
adds a distance and orders by it, browsing (``src/browse.py``) filters by
document and orders by ``chunk_id``. Each row joins the chunk with its
document's name and gathers its figure paths, and :func:`chunk_from_row` turns
it into a ``Chunk``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.dialects.postgresql import aggregate_order_by

from src.chunk import Chunk
from src.db.schema import chunks, documents, images


def stored_chunks_query(*extra_columns) -> Select:
    """Build the SELECT that reads stored chunks with their document and figures.

    One row per chunk: its record id, ``chunk_id``, original text, tables,
    summary, the document's ``source``, and ``image_paths``, the figure paths
    in position order (NULL when the chunk has none). Callers add their own
    filter, order and limit.

    Args:
        extra_columns: More columns to select, such as a distance.

    Returns:
        A SQLAlchemy ``Select``, run with ``connection.execute``.
    """
    image_paths = (
        select(func.array_agg(aggregate_order_by(images.c.path, images.c.position)))
        .where(images.c.chunk_id == chunks.c.id)
        .scalar_subquery()
        .label("image_paths")
    )
    return select(
        chunks.c.id,
        chunks.c.chunk_id,
        chunks.c.raw_text,
        chunks.c.tables_html,
        chunks.c.summary,
        documents.c.source,
        image_paths,
        *extra_columns,
    ).join(documents, chunks.c.document_id == documents.c.id)


def chunk_from_row(row: Any) -> Chunk:
    """Turn one row of :func:`stored_chunks_query` into a :class:`Chunk`.

    Args:
        row: A result row with the columns :func:`stored_chunks_query` selects.

    Returns:
        The chunk, with empty lists where the row has no tables or figures.
    """
    return Chunk(
        record_id=str(row.id),
        source=row.source,
        chunk_id=row.chunk_id,
        summary=row.summary,
        raw_text=row.raw_text or "",
        tables_html=list(row.tables_html or []),
        image_paths=[str(path) for path in row.image_paths or []],
    )
