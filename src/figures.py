"""Find a stored figure's file on disk, and only inside ``IMAGES_DIR``.

Figures are JPEG files under ``IMAGES_DIR`` (``dbv2/images``); the database
stores their paths, relative to the repository root. Two places read them:
the prompt (``src/retrieval/prompt.py``), which sends figures to the model,
and the image endpoint (``api/service.py:RAGService.image``), which serves
them. Both go through :func:`figure_file`, so a stored path that climbs out
of the images folder (``../``) is never read or served.
"""

from __future__ import annotations

from pathlib import Path

from config.settings import IMAGES_DIR
from src.chunk import Chunk
from src.logger import get_logger

logger = get_logger(__name__)

# Repository root: <repo>/src/figures.py -> <repo>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def figure_file(stored_path: str) -> Path | None:
    """Resolve a stored figure path to its file, only if it lies inside ``IMAGES_DIR``.

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


def chunk_figure_file(chunk: Chunk, position: int) -> Path | None:
    """Find the file of one of a chunk's figures.

    Args:
        chunk: The chunk.
        position: 0-based index into ``chunk.image_paths``.

    Returns:
        The absolute path of the figure, or None when the position is out of
        range or :func:`figure_file` rejects the path.
    """
    if not 0 <= position < len(chunk.image_paths):
        return None
    return figure_file(chunk.image_paths[position])
