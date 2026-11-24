"""Copy documents from the dev database to the production database, unchanged.

Documents are ingested into the Neon ``dev`` branch (``DATABASE_URL_DIRECT``)
and checked there. This script then copies them, row for row, into
``production`` (``PRODUCTION_DATABASE_URL_DIRECT``): the same record ids,
summaries and vectors, so no embedding or summary model is called. Each
document is copied in its own transaction (``src/db/store.py:copy_document``),
so a failure leaves the target's copy of that document as it was.

A document the target already holds is skipped unless ``--replace`` is given.
A document that fails is logged and the batch moves on.

The figure files are not in either database. Commit the new ones in
``dbv2/images/`` and redeploy, so the deployed API can serve them.

Usage:
    python -m scripts.copy_document lm317
    python -m scripts.copy_document lm317 ne555 --replace
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import Engine

from config.settings import DATABASE_URL_DIRECT, PRODUCTION_DATABASE_URL_DIRECT
from src.db.connection import create_db_engine
from src.db.store import copy_document
from src.logger import get_logger

logger = get_logger(__name__)


def _same_database(a: Engine, b: Engine) -> bool:
    """Whether two engines point at the same host and database."""
    return (a.url.host, a.url.port, a.url.database) == (b.url.host, b.url.port, b.url.database)


def main() -> int:
    """Parse arguments, copy each document, and return the process exit code."""
    parser = argparse.ArgumentParser(description="Copy documents from dev to production, unchanged.")
    parser.add_argument("sources", nargs="+", help="document names, such as lm317")
    parser.add_argument("--replace", action="store_true", help="replace a document production already holds")
    args = parser.parse_args()

    dev = create_db_engine(DATABASE_URL_DIRECT, "DATABASE_URL_DIRECT")
    production = create_db_engine(PRODUCTION_DATABASE_URL_DIRECT, "PRODUCTION_DATABASE_URL_DIRECT")
    if _same_database(dev, production):
        logger.error("DATABASE_URL_DIRECT and PRODUCTION_DATABASE_URL_DIRECT name the same database")
        return 1

    copied, skipped, failed = [], [], []
    for source in args.sources:
        try:
            copy_document(dev, production, source, replace=args.replace)
        except FileExistsError as exc:
            logger.info("Skipped: %s", exc)
            skipped.append(source)
        except Exception:
            logger.exception("Failed: %s", source)
            failed.append(source)
        else:
            copied.append(source)

    logger.info("Copied %d, skipped %d, failed %d", len(copied), len(skipped), len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
