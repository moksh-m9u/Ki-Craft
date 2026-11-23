"""Tests for the Chroma-to-Postgres copy and the database URL helper (no database needed)."""

import json
import uuid

import numpy as np
import pytest

from scripts.chroma_to_postgres import rows_from_chroma
from src.db.connection import sqlalchemy_url

ID_A = "004ca22b-a528-444e-9c0e-27c8b5858ff5"
ID_B = "11111111-2222-4333-8444-555555555555"
ID_C = "99999999-8888-4777-8666-555555555555"


def _records(*entries):
    """Build a Chroma ``get`` result from (id, source, chunk_id, tables, images) tuples."""
    return {
        "ids": [e[0] for e in entries],
        "documents": [f"summary of {e[1]} chunk {e[2]}" for e in entries],
        "metadatas": [
            {
                "source": e[1],
                "chunk_id": e[2],
                "raw_text": f"text {e[2]}",
                "tables_html": json.dumps(e[3]),
                "image_paths": json.dumps(e[4]),
                "has_image": bool(e[4]),
                "has_table": bool(e[3]),
            }
            for e in entries
        ],
        "embeddings": np.arange(len(entries) * 768, dtype=np.float32).reshape(len(entries), 768),
    }


def test_rows_keep_record_ids_and_group_by_document():
    records = _records(
        (ID_A, "lm317", 1, ["<table></table>"], ["dbv2/images/lm317_c1_img0.jpg", "dbv2/images/lm317_c1_img1.jpg"]),
        (ID_B, "lm317", 2, [], []),
        (ID_C, "ne555", 1, [], ["dbv2/images/ne555_c1_img0.jpg"]),
    )
    document_rows, chunk_rows, image_rows = rows_from_chroma(records)

    assert sorted(d["source"] for d in document_rows) == ["lm317", "ne555"]
    by_source = {d["source"]: d["id"] for d in document_rows}
    assert [c["id"] for c in chunk_rows] == [uuid.UUID(ID_A), uuid.UUID(ID_B), uuid.UUID(ID_C)]
    assert [c["document_id"] for c in chunk_rows] == [by_source["lm317"], by_source["lm317"], by_source["ne555"]]
    assert chunk_rows[0]["tables_html"] == ["<table></table>"]
    assert chunk_rows[1]["tables_html"] == []
    assert chunk_rows[0]["summary"] == "summary of lm317 chunk 1"
    assert np.array_equal(chunk_rows[2]["embedding"], records["embeddings"][2])
    assert image_rows == [
        {"chunk_id": uuid.UUID(ID_A), "position": 0, "path": "dbv2/images/lm317_c1_img0.jpg"},
        {"chunk_id": uuid.UUID(ID_A), "position": 1, "path": "dbv2/images/lm317_c1_img1.jpg"},
        {"chunk_id": uuid.UUID(ID_C), "position": 0, "path": "dbv2/images/ne555_c1_img0.jpg"},
    ]


def test_duplicate_chunk_in_a_document_is_refused():
    records = _records((ID_A, "lm317", 1, [], []), (ID_B, "lm317", 1, [], []))
    with pytest.raises(ValueError, match="lm317 has two records for chunk 1"):
        rows_from_chroma(records)


@pytest.mark.parametrize("scheme", ["postgresql://", "postgres://"])
def test_url_is_pointed_at_psycopg3(scheme):
    url = scheme + "user:pw@host-pooler.example/neondb?sslmode=require"
    assert sqlalchemy_url(url) == "postgresql+psycopg://user:pw@host-pooler.example/neondb?sslmode=require"


def test_empty_url_names_the_setting():
    with pytest.raises(RuntimeError, match="DATABASE_URL_DIRECT is not set"):
        sqlalchemy_url("", "DATABASE_URL_DIRECT")
