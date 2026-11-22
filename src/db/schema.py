"""Postgres tables that replace the Chroma store (Checkpoint 6).

Where it sits: ingestion will write these tables, retrieval will search
``chunks.embedding`` with pgvector, and ``src/browse.py`` will list documents
and chunks from them. For now only the migration in
``migrations/versions/`` uses this file, to create them.

The tables mirror what one Chroma record holds today:

    Chroma record                      Postgres
    id (UUID)                       -> chunks.id, unchanged, so figure URLs
                                       /v1/images/{record_id}/{position} keep working
    metadata.source                 -> documents.source
    metadata.chunk_id               -> chunks.chunk_id (1, 2, ... within a document)
    page_content (vision summary)   -> chunks.summary
    embedding of the summary        -> chunks.embedding, vector(768)
    metadata.raw_text               -> chunks.raw_text
    metadata.tables_html (JSON)     -> chunks.tables_html, text[]
    metadata.image_paths (JSON)     -> images, one row per figure

``has_image`` and ``has_table`` are not stored: they follow from ``images``
and ``tables_html``. Page numbers and bounding boxes arrive with Checkpoint 7.

SQLAlchemy Core is used, not its ORM: each ``Table`` is a plain description
of columns that queries are written against, with no classes or sessions.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID

# The watsonx Granite embedding model's output size (src/embed.py). A
# different embedding model means a new column and re-embedding every chunk.
EMBEDDING_DIMENSIONS = 768

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    # The PDF's file name without .pdf; the API's document_id in
    # /v1/documents/{document_id}, so it must stay unique and unchanged.
    Column("source", Text, nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

chunks = Table(
    "chunks",
    metadata,
    # No server default: every chunk keeps its Chroma record id, and new chunks
    # get theirs from ingestion, as they do today.
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("document_id", UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("chunk_id", Integer, nullable=False),
    Column("raw_text", Text, nullable=False, server_default=""),
    Column("tables_html", ARRAY(Text), nullable=False, server_default=text("'{}'")),
    Column("summary", Text, nullable=False),
    Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
    UniqueConstraint("document_id", "chunk_id"),
)

images = Table(
    "images",
    metadata,
    Column("chunk_id", UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True),
    # 0-based, the {position} in /v1/images/{record_id}/{position}.
    Column("position", Integer, primary_key=True),
    # Relative to the repository root, e.g. dbv2/images/lm317_c4_img0.jpg;
    # the image endpoint still serves only files inside IMAGES_DIR.
    Column("path", Text, nullable=False),
)
