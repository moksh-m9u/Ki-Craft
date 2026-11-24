"""Ingest PDFs, or existing chunk exports, into Postgres.

A PDF goes through :func:`src.ingestion.pipeline.run_complete_ingestion_pipeline`:
the chunk export goes to ``<exports>/<pdf name>.json``, figures to
``<images>``, and the chunks to the database at ``DATABASE_URL_DIRECT``. A PDF whose export already exists
is skipped, so an interrupted batch can be re-run with the same command.

A ``.json`` export goes through :func:`src.ingestion.pipeline.store_export`:
its summaries are embedded as they are, with no summary model. It is skipped
if the database already holds that document, unless ``--replace`` is given:
then the document's old chunks and figure files are replaced, after the new
summaries have been embedded. This is also how a new database is filled:
``alembic upgrade head``, then ``python -m scripts.ingest json/*.json``.

A file that fails is logged and the batch moves on; nothing of it is stored.

Usage:
    python -m scripts.ingest data/ne555.pdf
    python -m scripts.ingest data/ --backend local
    python -m scripts.ingest data/ne555.pdf --backend local --images /tmp/images --exports /tmp/json
    python -m scripts.ingest json/ne555.json
    python -m scripts.ingest json/lm317.json json/AN699.json --replace

The summary backend's tunables (timeout, retries, max tokens, image limits)
are environment variables; see the enhancement settings in config/settings.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config.settings import (
    ENHANCEMENT_BACKEND,
    ENHANCEMENT_BACKENDS,
    IMAGES_DIR,
    JSON_EXPORT_DIR,
)
from src.ingestion.enrich import check_enhancement_backend
from src.ingestion.pipeline import run_complete_ingestion_pipeline, store_export
from src.logger import get_logger

logger = get_logger(__name__)


def _inputs(paths: list[str]) -> list[Path]:
    """Expand folders to the PDFs inside them, sorted by name; keep files as given."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        found.extend(sorted(path.glob("*.pdf")) if path.is_dir() else [path])
    return found


def main() -> int:
    """Parse arguments, ingest each PDF, and return the process exit code."""
    parser = argparse.ArgumentParser(description="Ingest PDFs or chunk exports into Postgres.")
    parser.add_argument("paths", nargs="+", help="PDF files, folders of PDFs, or .json chunk exports")
    parser.add_argument("--backend", choices=sorted(ENHANCEMENT_BACKENDS), default=ENHANCEMENT_BACKEND,
                        help="summary model backend (default: ENHANCEMENT_BACKEND, %(default)s)")
    parser.add_argument("--images", default=IMAGES_DIR, help="figure files folder (default: %(default)s)")
    parser.add_argument("--exports", default=JSON_EXPORT_DIR, help="chunk exports folder (default: %(default)s)")
    parser.add_argument("--replace", action="store_true",
                        help="for .json exports: replace a document the database already holds")
    args = parser.parse_args()

    inputs = _inputs(args.paths)
    if any(path.suffix.lower() != ".json" for path in inputs):
        check_enhancement_backend(args.backend)
    logger.info("Ingesting %d files with the %s backend", len(inputs), args.backend)

    done, skipped, failed = [], [], []
    for n, pdf in enumerate(inputs, start=1):
        logger.info("[%d/%d] %s", n, len(inputs), pdf)
        t0 = time.monotonic()
        try:
            if pdf.suffix.lower() == ".json":
                store_export(pdf, args.images, replace=args.replace)
            else:
                run_complete_ingestion_pipeline(pdf, args.images, args.exports, args.backend)
        except FileExistsError as exc:
            logger.info("  Skipped: %s", exc)
            skipped.append(pdf)
            continue
        except Exception:
            logger.exception("  Failed: %s", pdf)
            failed.append(pdf)
            continue
        logger.info("  Done in %.0fs", time.monotonic() - t0)
        done.append(pdf)

    logger.info("Ingested %d, skipped %d, failed %d", len(done), len(skipped), len(failed))
    for pdf in failed:
        logger.info("  failed: %s", pdf)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
