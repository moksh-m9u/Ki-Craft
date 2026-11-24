"""Read and write one whole document's rows: the ``documents`` row, its chunks and figures.

Where it sits: ingestion (``src/ingestion/pipeline.py:store_export``) writes a
document it has just embedded; ``scripts/copy_document.py`` reads a document
from one database (the Neon ``dev`` branch) and writes it, unchanged, to
another (``production``). Both write through :func:`write_document`, so a
document is always stored the same way: whole, in one transaction, or not at
all.

A row here is a plain dict keyed by column name, ready for SQLAlchemy's
``insert``: ``chunk_rows`` have ``id``, ``chunk_id``, ``raw_text``,
``tables_html``, ``summary`` and ``embedding``; ``image_rows`` have
``chunk_id`` (the chunk's record id), ``position`` and ``path``.
"""

from __future__ import annotations

from sqlalchemy import Engine, delete, select

from src.db.schema import chunks, documents, images
from src.logger import get_logger

logger = get_logger(__name__)

_CHUNK_COLUMNS = ("id", "chunk_id", "raw_text", "tables_html", "summary", "embedding")


def is_stored(engine: Engine, source: str) -> bool:
    """Whether the database already holds a document named ``source``."""
    with engine.connect() as connection:
        return connection.execute(select(documents.c.id).where(documents.c.source == source)).first() is not None


def write_document(engine: Engine, source: str, chunk_rows: list[dict], image_rows: list[dict]) -> None:
    """Store one document's rows in one transaction, replacing any it already has.

    Deleting the ``documents`` row deletes its chunks and their figure rows
    too (``ON DELETE CASCADE``). If an insert fails, the transaction rolls
    back and the old document is still there.

    Args:
        engine: The database to write to.
        source: The document's name, such as ``lm317``.
        chunk_rows: One dict per chunk, without ``document_id`` (added here).
        image_rows: One dict per figure.
    """
    with engine.begin() as connection:
        connection.execute(delete(documents).where(documents.c.source == source))
        document_id = connection.execute(
            documents.insert().values(source=source).returning(documents.c.id)
        ).scalar_one()
        connection.execute(chunks.insert(), [{**row, "document_id": document_id} for row in chunk_rows])
        if image_rows:
            connection.execute(images.insert(), image_rows)


def read_document(engine: Engine, source: str) -> tuple[list[dict], list[dict]] | None:
    """Read one document's chunk and figure rows exactly as stored, vectors included.

    Args:
        engine: The database to read from.
        source: The document's name.

    Returns:
        ``(chunk_rows, image_rows)`` in the shape :func:`write_document`
        takes, chunks in reading order; None when the database has no such
        document.
    """
    document_id = select(documents.c.id).where(documents.c.source == source).scalar_subquery()
    chunk_query = (
        select(*(chunks.c[name] for name in _CHUNK_COLUMNS))
        .where(chunks.c.document_id == document_id)
        .order_by(chunks.c.chunk_id)
    )
    image_query = (
        select(images.c.chunk_id, images.c.position, images.c.path)
        .join(chunks, images.c.chunk_id == chunks.c.id)
        .where(chunks.c.document_id == document_id)
        .order_by(chunks.c.chunk_id, images.c.position)
    )
    with engine.connect() as connection:
        if not connection.execute(select(document_id)).scalar():
            return None
        chunk_rows = [dict(row._mapping) for row in connection.execute(chunk_query)]
        image_rows = [dict(row._mapping) for row in connection.execute(image_query)]
    return chunk_rows, image_rows


def copy_document(from_engine: Engine, to_engine: Engine, source: str, replace: bool = False) -> int:
    """Copy one document from one database to another without changing it.

    The rows are copied as they are: the same record ids (so figure URLs mean
    the same figure in both databases), summaries and vectors. No embedding
    or summary model is called. The figure files are not copied; they live
    on disk (``IMAGES_DIR``) and both databases point at the same paths.

    Args:
        from_engine: The database to read from (the ``dev`` branch).
        to_engine: The database to write to (the ``production`` branch).
        source: The document's name.
        replace: Replace the document if the target already holds it.

    Returns:
        The number of chunks copied.

    Raises:
        LookupError: If the source database has no such document.
        FileExistsError: If the target already holds it and ``replace`` is
            false; nothing is changed.
    """
    rows = read_document(from_engine, source)
    if rows is None:
        raise LookupError(f"the source database has no document named {source}")
    if is_stored(to_engine, source) and not replace:
        raise FileExistsError(f"the target database already holds {source}; pass --replace to replace it")
    chunk_rows, image_rows = rows
    write_document(to_engine, source, chunk_rows, image_rows)
    logger.info("Copied %s: %d chunks, %d figures", source, len(chunk_rows), len(image_rows))
    return len(chunk_rows)
