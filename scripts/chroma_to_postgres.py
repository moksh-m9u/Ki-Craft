"""Copy every record from the Chroma store into Postgres, once (Checkpoint 6).

Reads the ids, summaries, embeddings and metadata Chroma holds and writes
them to the ``documents``, ``chunks`` and ``images`` tables created by the
first migration. Nothing is summarized or embedded again: each chunk keeps
its Chroma record id and its exact embedding, so figure URLs keep working and
pgvector can be checked against Chroma question by question.

The copy is one transaction: either every row is written or none is. It
refuses to run when ``chunks`` already holds rows. To copy again, empty the
tables first:
    alembic downgrade base
    alembic upgrade head

After writing, it reads every row back and checks it against Chroma: the
same ids, documents, chunk numbers, text, tables, figure paths, and
bit-for-bit the same embeddings.

Opening Chroma rewrites tracked files under dbv2/chroma_db; run
``git checkout -- dbv2/chroma_db`` afterwards.

Usage:
    python -m scripts.chroma_to_postgres
    python -m scripts.chroma_to_postgres --check-only
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

import chromadb
import numpy as np
from sqlalchemy import Engine, create_engine, func, select

from config.settings import CHROMA_PERSIST_DIR, DATABASE_URL_DIRECT
from src.db.connection import sqlalchemy_url
from src.db.schema import chunks, documents, images
from src.logger import get_logger

logger = get_logger(__name__)

# The collection name LangChain's Chroma wrapper uses when none is given.
COLLECTION = "langchain"


def read_chroma(persist_directory: str = CHROMA_PERSIST_DIR) -> dict[str, Any]:
    """Read every record of the Chroma collection: ids, summaries, metadata and embeddings.

    Uses the chromadb client directly, so no embedding model (and no watsonx
    call) is needed.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_collection(COLLECTION).get(include=["documents", "metadatas", "embeddings"])


def rows_from_chroma(records: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    """Turn a Chroma ``get`` result into rows for ``documents``, ``chunks`` and ``images``.

    Chroma stores lists as JSON strings in metadata (``tables_html``,
    ``image_paths``); they become a ``text[]`` column and one ``images`` row
    per figure. Each document gets a new UUID; each chunk keeps its record id.

    Args:
        records: The dict ``collection.get(include=[documents, metadatas, embeddings])`` returns.

    Returns:
        ``(document_rows, chunk_rows, image_rows)``, ready to insert.

    Raises:
        ValueError: If two records share a document and chunk number, which the
            unique constraint would reject halfway through the copy.
    """
    document_ids: dict[str, uuid.UUID] = {}
    chunk_rows: list[dict] = []
    image_rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for record_id, summary, meta, embedding in zip(
        records["ids"], records["documents"], records["metadatas"], records["embeddings"], strict=True
    ):
        source = meta["source"]
        key = (source, int(meta["chunk_id"]))
        if key in seen:
            raise ValueError(f"{source} has two records for chunk {key[1]}")
        seen.add(key)
        document_id = document_ids.setdefault(source, uuid.uuid4())
        chunk_rows.append(
            {
                "id": uuid.UUID(record_id),
                "document_id": document_id,
                "chunk_id": key[1],
                "raw_text": meta.get("raw_text") or "",
                "tables_html": json.loads(meta.get("tables_html") or "[]"),
                "summary": summary,
                "embedding": np.asarray(embedding, dtype=np.float32),
            }
        )
        for position, path in enumerate(json.loads(meta.get("image_paths") or "[]")):
            image_rows.append({"chunk_id": uuid.UUID(record_id), "position": position, "path": path})
    document_rows = [{"id": document_id, "source": source} for source, document_id in document_ids.items()]
    return document_rows, chunk_rows, image_rows


def copy(engine: Engine, document_rows: list[dict], chunk_rows: list[dict], image_rows: list[dict]) -> None:
    """Insert all rows in one transaction.

    Raises:
        RuntimeError: If ``chunks`` already holds rows.
    """
    with engine.begin() as connection:
        existing = connection.execute(select(func.count()).select_from(chunks)).scalar_one()
        if existing:
            raise RuntimeError(
                f"chunks already holds {existing} rows; run `alembic downgrade base` and "
                "`alembic upgrade head` to empty the tables, then copy again"
            )
        connection.execute(documents.insert(), document_rows)
        connection.execute(chunks.insert(), chunk_rows)
        if image_rows:
            connection.execute(images.insert(), image_rows)


def check(engine: Engine, records: dict[str, Any]) -> list[str]:
    """Read Postgres back and compare it with the Chroma records.

    Returns:
        One line per difference; an empty list means Postgres holds exactly
        what Chroma holds.
    """
    _, expected_chunks, expected_images = rows_from_chroma(records)
    with engine.connect() as connection:
        stored = {
            row.id: row
            for row in connection.execute(
                select(chunks, documents.c.source).join(documents, chunks.c.document_id == documents.c.id)
            )
        }
        stored_images = {(row.chunk_id, row.position): row.path for row in connection.execute(select(images))}
    sources = dict(zip((uuid.UUID(i) for i in records["ids"]), (m["source"] for m in records["metadatas"])))

    problems: list[str] = []
    if set(stored) != {row["id"] for row in expected_chunks}:
        problems.append(f"chunk ids differ: Postgres {len(stored)}, Chroma {len(expected_chunks)}")
    for row in expected_chunks:
        got = stored.get(row["id"])
        if got is None:
            continue
        for column in ("chunk_id", "raw_text", "tables_html", "summary"):
            if getattr(got, column) != row[column]:
                problems.append(f"{row['id']}: {column} differs")
        if got.source != sources[row["id"]]:
            problems.append(f"{row['id']}: document differs")
        if not np.array_equal(np.asarray(got.embedding, dtype=np.float32), row["embedding"]):
            problems.append(f"{row['id']}: embedding differs")
    if stored_images != {(r["chunk_id"], r["position"]): r["path"] for r in expected_images}:
        problems.append(f"images differ: Postgres {len(stored_images)}, Chroma {len(expected_images)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Copy Chroma into Postgres, then check the copy. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-only", action="store_true", help="compare without copying")
    args = parser.parse_args(argv)

    records = read_chroma()
    engine = create_engine(sqlalchemy_url(DATABASE_URL_DIRECT, "DATABASE_URL_DIRECT"))
    if not args.check_only:
        document_rows, chunk_rows, image_rows = rows_from_chroma(records)
        copy(engine, document_rows, chunk_rows, image_rows)
        print(f"Copied {len(document_rows)} documents, {len(chunk_rows)} chunks, {len(image_rows)} figures")

    problems = check(engine, records)
    for problem in problems:
        print(problem)
    print("Postgres matches Chroma" if not problems else f"{len(problems)} differences")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
