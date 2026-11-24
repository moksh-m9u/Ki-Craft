"""Tests for src/documents.py: reading a stored chunk's metadata."""

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.documents import Document

import src.documents as documents
from src.documents import chunk_text_and_tables, image_file, stored_image_count


def _doc(metadata):
    return Document(page_content="summary", metadata=metadata)


def test_text_and_tables_are_read_from_metadata():
    doc = _doc({"raw_text": "flat", "tables_html": ["<t/>"], "image_paths": []})
    assert chunk_text_and_tables(doc) == {"raw_text": "flat", "tables_html": ["<t/>"]}


def test_metadata_without_optional_keys():
    assert chunk_text_and_tables(_doc({})) == {"raw_text": "", "tables_html": []}


def test_figure_count_reads_no_file():
    doc = _doc({"image_paths": ["dbv2/images/missing_a.jpg", "dbv2/images/missing_b.jpg"]})
    assert stored_image_count(doc) == 2


def test_image_file_finds_a_figure_inside_the_images_folder(tmp_path, monkeypatch):
    (tmp_path / "dbv2" / "images").mkdir(parents=True)
    (tmp_path / "dbv2" / "images" / "fig.jpg").write_bytes(b"jpeg-bytes")
    monkeypatch.setattr(documents, "_PROJECT_ROOT", tmp_path)
    metadata = {"image_paths": ["dbv2/images/fig.jpg", "dbv2/images/gone.jpg", "secret.txt"]}
    assert image_file(metadata, 0) == tmp_path / "dbv2" / "images" / "fig.jpg"
    assert image_file(metadata, 1) is None, "a missing file"
    assert image_file(metadata, 2) is None, "a path outside the images folder"
    assert image_file(metadata, 3) is None, "a position past the end"
