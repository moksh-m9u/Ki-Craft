"""Create documents, chunks and images, and enable pgvector.

The first schema: the three tables that replace the Chroma store. Their
meaning is described in ``src/db/schema.py``. A migration spells its tables
out instead of importing that file, so it keeps creating exactly this schema
after ``schema.py`` changes in later migrations.

Revision ID: 0001
Revises:
Create Date: 2026-11-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    """Enable pgvector and create the three tables."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source", sa.Text(), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("tables_html", ARRAY(sa.Text()), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_id"),
    )
    op.create_table(
        "images",
        sa.Column("chunk_id", UUID(as_uuid=True), sa.ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("position", sa.Integer(), primary_key=True),
        sa.Column("path", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    """Drop the three tables. The vector extension stays, as other tables may use it."""
    op.drop_table("images")
    op.drop_table("chunks")
    op.drop_table("documents")
