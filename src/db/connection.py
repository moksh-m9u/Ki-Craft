"""Opens Postgres: turns a Neon connection string into a SQLAlchemy engine.

``create_db_engine`` is what the API, retrieval, ingestion and the scripts
use; ``migrations/env.py`` uses ``sqlalchemy_url`` directly, because Alembic
opens its own single connection.

An engine is SQLAlchemy's connection pool: it opens connections when asked,
keeps a few open for reuse, and is created once per process.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, text


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


def create_db_engine(url: str, name: str = "DATABASE_URL") -> Engine:
    """Create the engine (connection pool) for one Neon connection string.

    Two settings make it safe behind Neon's pooler and on Render:
    ``prepare_threshold=None`` stops psycopg from creating server-side
    prepared statements, which a transaction pooler (PgBouncer) cannot keep
    across transactions; ``pool_pre_ping`` tests a pooled connection before
    using it, because Neon closes idle connections when it scales to zero and
    Render can reset outbound connections.

    Args:
        url: ``DATABASE_URL`` (pooled) or ``DATABASE_URL_DIRECT``.
        name: The setting it came from, for the error when it is empty.

    Returns:
        The engine. No connection is opened until it is first used.

    Raises:
        RuntimeError: If the URL is empty.
    """
    return create_engine(
        sqlalchemy_url(url, name),
        pool_pre_ping=True,
        connect_args={"prepare_threshold": None},
    )


def ping(engine: Engine) -> None:
    """Run ``SELECT 1``, so a wrong URL or an unreachable database fails at startup.

    The first query after Neon has scaled to zero also wakes it, which takes
    about a second.

    Raises:
        sqlalchemy.exc.OperationalError: If the database cannot be reached.
    """
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
