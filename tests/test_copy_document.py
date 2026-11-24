"""Tests for src/db/store.py:copy_document and scripts/copy_document.py.

The SQL functions (``read_document``, ``is_stored``, ``write_document``) are
replaced by two in-memory databases, so no test needs Postgres; their SQL is
checked live (see AGENTS.md, Checkpoint 6).
"""

from __future__ import annotations

import uuid

import pytest

from src.db import store
from src.db.store import copy_document


class _Databases:
    """Two in-memory databases, keyed by engine name, each mapping source -> rows."""

    def __init__(self, monkeypatch):
        self.data: dict[str, dict[str, tuple[list[dict], list[dict]]]] = {"dev": {}, "production": {}}
        monkeypatch.setattr(store, "read_document", lambda engine, source: self.data[engine].get(source))
        monkeypatch.setattr(store, "is_stored", lambda engine, source: source in self.data[engine])
        monkeypatch.setattr(store, "write_document", self.write)

    def write(self, engine, source, chunk_rows, image_rows):
        """Replace the document, as the one transaction would."""
        self.data[engine][source] = (chunk_rows, image_rows)


def _rows(summary: str = "summary") -> tuple[list[dict], list[dict]]:
    """One chunk with one figure, as read_document returns them."""
    record_id = uuid.uuid4()
    chunk = {"id": record_id, "chunk_id": 1, "raw_text": "text", "tables_html": [], "summary": summary,
             "embedding": [0.1, 0.2]}
    return [chunk], [{"chunk_id": record_id, "position": 0, "path": "dbv2/images/lm317_c1_img0.jpg"}]


@pytest.fixture
def databases(monkeypatch) -> _Databases:
    """Empty dev and production databases."""
    return _Databases(monkeypatch)


def test_copy_keeps_record_ids_vectors_and_figures(databases):
    databases.data["dev"]["lm317"] = _rows()
    assert copy_document("dev", "production", "lm317") == 1
    assert databases.data["production"]["lm317"] == databases.data["dev"]["lm317"]


def test_copy_refuses_a_document_the_target_holds(databases):
    databases.data["dev"]["lm317"] = _rows("new")
    databases.data["production"]["lm317"] = old = _rows("old")
    with pytest.raises(FileExistsError, match="--replace"):
        copy_document("dev", "production", "lm317")
    assert databases.data["production"]["lm317"] is old


def test_replace_overwrites_the_target(databases):
    databases.data["dev"]["lm317"] = new = _rows("new")
    databases.data["production"]["lm317"] = _rows("old")
    copy_document("dev", "production", "lm317", replace=True)
    assert databases.data["production"]["lm317"] == new


def test_unknown_document_is_an_error_and_writes_nothing(databases):
    with pytest.raises(LookupError, match="no document named nope"):
        copy_document("dev", "production", "nope")
    assert databases.data["production"] == {}


def test_script_refuses_to_copy_a_database_onto_itself(monkeypatch):
    import sys

    import scripts.copy_document as script
    from sqlalchemy import create_engine

    url = "postgresql+psycopg://user@db.example/neondb"
    monkeypatch.setattr(script, "create_db_engine", lambda url_, name: create_engine(url))
    monkeypatch.setattr(script, "copy_document", lambda *a, **k: pytest.fail("must not copy"))
    monkeypatch.setattr(sys, "argv", ["copy_document", "lm317"])
    assert script.main() == 1
