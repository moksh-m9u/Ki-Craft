"""Turns a Neon connection string into one SQLAlchemy can open.

Used by ``migrations/env.py`` and ``scripts/chroma_to_postgres.py``. The API
will use it too once it reads from Postgres.
"""

from __future__ import annotations


def sqlalchemy_url(url: str, name: str = "DATABASE_URL") -> str:
    """Point a ``postgresql://`` URL at the psycopg 3 driver.

    Neon gives plain ``postgresql://`` URLs, which SQLAlchemy would hand to
    psycopg2, a driver this project does not install.

    Args:
        url: The connection string from ``config/settings.py``.
        name: The setting it came from, named in the error when it is empty.

    Returns:
        The same URL starting with ``postgresql+psycopg://``.

    Raises:
        RuntimeError: If the URL is empty (the setting is not in ``.env``).
    """
    if not url:
        raise RuntimeError(f"{name} is not set; add the Neon connection string to .env")
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url
