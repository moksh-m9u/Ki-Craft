"""Read a stored chunk's original text, tables and figures from its metadata.

A chunk's ``page_content`` is its summary; its ``metadata`` holds the original
content: ``raw_text``, ``tables_html`` and ``image_paths`` (lists; JSON strings,
as the old Chroma store kept them, are still accepted). Figures are files under
``IMAGES_DIR``; ``image_paths`` lists them in order.

Functions:

* :func:`stored_chunks_query`, :func:`document_from_row` - read chunks from
  Postgres as LangChain ``Document`` objects, the shape retrieval, the prompt
  and the API already use.

* :func:`chunk_text_and_tables` - text and tables, no figure read.
* :func:`stored_image_paths`, :func:`stored_image_count` - which figures a
  chunk has, without reading them.
* :func:`image_file`, :func:`figure_file` - the file behind one figure, kept
  inside ``IMAGES_DIR``; the model prompt and the image endpoint both find
  figures through :func:`figure_file`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import Select, func, select
from sqlalchemy.dialects.postgresql import aggregate_order_by

from config.settings import IMAGES_DIR
from src.db.schema import chunks, documents, images
from src.logger import get_logger

logger = get_logger(__name__)

# Repository root: <repo>/src/documents.py -> <repo>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def document_from_row(row: Any) -> Document:
    """Turn one row of :func:`stored_chunks_query` into a LangChain ``Document``.

    The summary is ``page_content`` and the metadata keys are the ones Chroma
    held, so the prompt, citations and API code read it unchanged.
    """
    tables = list(row.tables_html or [])
    paths = list(row.image_paths or [])
    return Document(
        id=str(row.id),
        page_content=row.summary,
        metadata={
            "source": row.source,
            "chunk_id": row.chunk_id,
            "raw_text": row.raw_text,
            "tables_html": tables,
            "image_paths": paths,
            "has_table": bool(tables),
            "has_image": bool(paths),
        },
    )


def _parse_json_value(raw: Any, default: Any) -> Any:
    """Parse ``raw`` as JSON when it is a string, otherwise return it as-is."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return raw if raw is not None else default


def chunk_text_and_tables(chunk: Document) -> dict:
    """Return a chunk's original text and tables without reading its figures.

    Args:
        chunk: A retrieved LangChain ``Document``.

    Returns:
        Dict with ``raw_text`` (str) and ``tables_html`` (list[str]).
    """
    metadata = chunk.metadata or {}
    return {
        "raw_text": metadata.get("raw_text", ""),
        "tables_html": _parse_json_value(metadata.get("tables_html"), []),
    }


def stored_image_count(chunk: Document) -> int:
    """Count a chunk's figures without reading any."""
    return len(stored_image_paths(chunk.metadata or {}))


def stored_image_paths(metadata: dict) -> list[str]:
    """Return a record's figure paths in stored order."""
    paths = _parse_json_value(metadata.get("image_paths"), [])
    return [str(path) for path in paths] if isinstance(paths, list) else []


def image_file(metadata: dict, position: int) -> Path | None:
    """Find the file of one of a retrieved chunk's figures, for the model prompt.

    Args:
        metadata: The chunk's metadata, with ``image_paths``.
        position: 0-based index into ``image_paths``.

    Returns:
        The absolute path of the figure, or None when the position is out of
        range or :func:`figure_file` rejects the path.
    """
    paths = stored_image_paths(metadata)
    if not 0 <= position < len(paths):
        return None
    return figure_file(paths[position])


def figure_file(stored_path: str) -> Path | None:
    """Resolve a stored figure path to its file, only if it lies inside ``IMAGES_DIR``.

    The image endpoint and the prompt both go through here, so a stored path
    that climbs out of the images folder (``../``) is never read or served.

    Args:
        stored_path: A path as ingestion stored it, relative to the repository root.

    Returns:
        The absolute path, or None when it points outside ``IMAGES_DIR`` or the
        file is missing.
    """
    images_root = (_PROJECT_ROOT / IMAGES_DIR).resolve()
    full_path = (_PROJECT_ROOT / stored_path).resolve()
    if not full_path.is_relative_to(images_root) or not full_path.is_file():
        logger.warning("Stored figure path is outside %s or missing: %s", IMAGES_DIR, stored_path)
        return None
    return full_path
