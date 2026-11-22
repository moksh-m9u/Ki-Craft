"""Tests for the Postgres schema and its first migration (no database needed)."""

import importlib.util
import io
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from src.db import schema

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "versions" / "0001_documents_chunks_images.py"


def _migration_sql(direction: str) -> str:
    """Run the first migration's upgrade or downgrade in offline mode and return its SQL."""
    spec = importlib.util.spec_from_file_location("migration_0001", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer}
    )
    with Operations.context(context):
        getattr(module, direction)()
    return buffer.getvalue()


def test_tables_and_columns():
    assert {name: [c.name for c in table.columns] for name, table in schema.metadata.tables.items()} == {
        "documents": ["id", "source", "created_at"],
        "chunks": ["id", "document_id", "chunk_id", "raw_text", "tables_html", "summary", "embedding"],
        "images": ["chunk_id", "position", "path"],
    }


def test_embedding_matches_the_watsonx_model():
    assert schema.chunks.c.embedding.type.dim == 768 == schema.EMBEDDING_DIMENSIONS


def test_chunk_ids_come_from_chroma_not_the_database():
    # Figure URLs carry the Chroma record id, so the database must not invent one.
    assert schema.chunks.c.id.server_default is None
    assert schema.documents.c.id.server_default is not None


def test_deleting_a_document_deletes_its_chunks_and_images():
    (chunk_fk,) = schema.chunks.c.document_id.foreign_keys
    (image_fk,) = schema.images.c.chunk_id.foreign_keys
    assert (chunk_fk.column.table.name, chunk_fk.ondelete) == ("documents", "CASCADE")
    assert (image_fk.column.table.name, image_fk.ondelete) == ("chunks", "CASCADE")


def test_migration_creates_exactly_the_schema():
    sql = " ".join(_migration_sql("upgrade").split())
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    for table in schema.metadata.sorted_tables:
        # Alembic indents with spaces and SQLAlchemy with tabs, so compare words only.
        expected = " ".join(str(CreateTable(table).compile(dialect=postgresql.dialect())).split())
        assert expected in sql, f"migration and schema.py disagree on {table.name}"


def test_migration_downgrade_drops_the_tables_children_first():
    sql = _migration_sql("downgrade")
    drops = [line for line in sql.splitlines() if line.startswith("DROP TABLE")]
    assert drops == ["DROP TABLE images;", "DROP TABLE chunks;", "DROP TABLE documents;"]
