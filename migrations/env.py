"""Alembic environment: connects to Postgres and runs the migrations.

Alembic runs this file for every command (``alembic upgrade head`` and so
on). It opens one connection with ``DATABASE_URL_DIRECT`` (the direct Neon
host, not the pooler, because a migration is one long session) and runs the
scripts in ``migrations/versions/`` in order. ``target_metadata`` is the
schema in ``src/db/schema.py``, which ``alembic check`` compares with the
live database to catch a table that drifted from its description.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from config.settings import DATABASE_URL_DIRECT
from src.db.connection import sqlalchemy_url
from src.db.schema import metadata

target_metadata = metadata


def run_migrations_offline() -> None:
    """Print the SQL instead of running it (``alembic upgrade head --sql``)."""
    context.configure(url="postgresql+psycopg://", target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run the migrations against the database in one transaction."""
    engine = create_engine(sqlalchemy_url(DATABASE_URL_DIRECT, "DATABASE_URL_DIRECT"), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
