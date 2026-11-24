"""Tests for src/figures.py: a figure's file is found only inside the images folder."""

import src.figures as figures
from src.chunk import Chunk
from src.figures import chunk_figure_file


def test_chunk_figure_file_finds_a_figure_inside_the_images_folder(tmp_path, monkeypatch):
    (tmp_path / "dbv2" / "images").mkdir(parents=True)
    (tmp_path / "dbv2" / "images" / "fig.jpg").write_bytes(b"jpeg-bytes")
    monkeypatch.setattr(figures, "_PROJECT_ROOT", tmp_path)
    chunk = Chunk(
        record_id=None,
        source="lm317",
        chunk_id=1,
        summary="summary",
        image_paths=["dbv2/images/fig.jpg", "dbv2/images/gone.jpg", "secret.txt"],
    )
    assert chunk_figure_file(chunk, 0) == tmp_path / "dbv2" / "images" / "fig.jpg"
    assert chunk_figure_file(chunk, 1) is None, "a missing file"
    assert chunk_figure_file(chunk, 2) is None, "a path outside the images folder"
    assert chunk_figure_file(chunk, 3) is None, "a position past the end"
