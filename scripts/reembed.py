"""Build the configured embedding model's collection from the stored summaries.

The store keeps one collection per embedding model. This copies every record of
an existing collection (by default ``langchain``, the original Granite one)
into the configured model's collection, embedding each summary with that model
and keeping the record ids, summaries and metadata. See
:mod:`src.ingestion.reembed` for how a run resumes and when the new collection
becomes visible.

API embeddings send every summary to the provider (OpenRouter or watsonx).

Usage:
    python -m scripts.reembed --status
    EMBEDDING_BACKEND=openrouter python -m scripts.reembed --dry-run
    EMBEDDING_BACKEND=openrouter python -m scripts.reembed
    EMBEDDING_BACKEND=openrouter OPENROUTER_EMBEDDING_MODEL=baai/bge-m3 python -m scripts.reembed --store /tmp/trial/chroma_db

``--status`` makes no network call. ``--dry-run`` reports the records, batches
and estimated tokens a run would send, without sending them.
"""

from __future__ import annotations

import argparse
import sys

from config.settings import CHROMA_PERSIST_DIR, EMBEDDING_BACKEND, OPENROUTER_EMBEDDING_BATCH_SIZE
from src.embed import GRANITE_COLLECTION, collection_name, embedding_model_id, load_embedding_model
from src.ingestion.reembed import describe_collections, reembed_collection
from src.logger import get_logger

logger = get_logger(__name__)


def _print_status(persist_directory: str) -> None:
    """Print every collection in the store and whether the configured one exists."""
    rows = describe_collections(persist_directory)
    configured = collection_name()
    print(f"Store: {persist_directory}")
    print(f"Configured: EMBEDDING_BACKEND={EMBEDDING_BACKEND}, model {embedding_model_id()}, collection {configured}")
    for row in rows:
        model = row["embedding_model"] or ("granite (unrecorded)" if row["name"] == GRANITE_COLLECTION else "unrecorded")
        mark = "  <- configured" if row["configured"] else ""
        print(f"  {row['name']:40} {row['records']:5} records  {row['dimension'] or '-':>5} dims  {model}{mark}")
    if not any(row["configured"] for row in rows):
        print(f"  The configured collection {configured!r} does not exist yet: run python -m scripts.reembed")


def main() -> int:
    """Parse arguments, report or build the collection, and return the exit code."""
    parser = argparse.ArgumentParser(description="Build the configured embedding model's collection from stored summaries.")
    parser.add_argument("--store", default=CHROMA_PERSIST_DIR, help="Chroma folder (default: %(default)s)")
    parser.add_argument("--from", dest="source", default=GRANITE_COLLECTION,
                        help="collection to copy from (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=OPENROUTER_EMBEDDING_BATCH_SIZE,
                        help="records per embedding call (default: OPENROUTER_EMBEDDING_BATCH_SIZE, %(default)s)")
    parser.add_argument("--status", action="store_true", help="list the store's collections; no network call")
    parser.add_argument("--dry-run", action="store_true", help="report what a run would embed; no network call")
    args = parser.parse_args()

    if args.status:
        _print_status(args.store)
        return 0

    try:
        embedding = None if args.dry_run else load_embedding_model()
        report = reembed_collection(embedding, args.store, args.source, args.batch_size, dry_run=args.dry_run)
    except Exception as exc:
        # The batches already written stay in <target>-partial, so the same
        # command resumes from there.
        logger.error("Re-embedding failed: %s: %s", type(exc).__name__, exc)
        logger.error("Batches already embedded are kept; run the same command again to resume.")
        return 1
    print(f"{report['status']}: {report['source']} -> {report['target']} ({report['model']})")
    print(f"  records {report['records']}, embedded now {report['embedded']}, batches {report['batches']}, "
          f"estimated tokens {report['estimated_tokens']}")
    print(f"  figure references kept {report['figures']}, vector dimension {report['dimension'] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
