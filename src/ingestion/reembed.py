"""Copy a stored collection into another embedding model's collection.

The store keeps one Chroma collection per embedding model
(``src/embed.py:collection_name``). Switching models therefore means building
the new model's collection, and everything it needs is already in the store:
each record's summary (what gets embedded) and its metadata (``source``,
``chunk_id``, ``raw_text``, ``tables_html``, ``image_paths`` ...). No PDF is
parsed and no summary is written again.

:func:`reembed_collection` reads every record of the source collection, embeds
its summary with the configured model, and writes it under the same record id
with the same summary and metadata. Keeping the ids keeps figure URLs
(``/v1/images/{record_id}/{position}``) valid whichever collection the API
serves, and the figure files themselves are never touched.

The records are written to ``<target>-partial`` one batch at a time, so an
interrupted run (a rate limit, a network error) resumes where it stopped
without paying for the earlier batches again. Only when the copy matches the
source record for record is it renamed to ``<target>``; until then the API
never sees it.
"""

from __future__ import annotations

import json
import math

import chromadb
from chromadb.errors import NotFoundError
from langchain_core.embeddings import Embeddings

from config.settings import CHROMA_PERSIST_DIR, OPENROUTER_EMBEDDING_BATCH_SIZE
from src.embed import GRANITE_COLLECTION, collection_metadata, collection_name, embedding_model_id
from src.logger import get_logger
from src.retrieval.search import check_collection

logger = get_logger(__name__)

PARTIAL_SUFFIX = "-partial"


def describe_collections(persist_directory: str = CHROMA_PERSIST_DIR) -> list[dict]:
    """List every collection in the store with its model, size and dimension.

    Makes no network call.

    Args:
        persist_directory: Folder of the Chroma store.

    Returns:
        One dict per collection: ``name``, ``embedding_model`` (``None`` when
        unrecorded), ``records``, ``dimension`` (``None`` when empty) and
        ``configured`` (whether the current settings use it).
    """
    client = chromadb.PersistentClient(path=persist_directory)
    configured = collection_name()
    rows = []
    for collection in sorted(client.list_collections(), key=lambda c: c.name):
        sample = collection.get(limit=1, include=["embeddings"])["embeddings"]
        rows.append(
            {
                "name": collection.name,
                "embedding_model": (collection.metadata or {}).get("embedding_model"),
                "records": collection.count(),
                "dimension": len(sample[0]) if len(sample) else None,
                "configured": collection.name == configured,
            }
        )
    return rows


def _verify(source: dict, copy) -> int:
    """Return the copy's vector dimension; raise RuntimeError unless it holds exactly the source's records."""
    got = copy.get(include=["documents", "metadatas", "embeddings"])
    if sorted(got["ids"]) != sorted(source["ids"]):
        raise RuntimeError(f"copy holds {len(got['ids'])} records, the source {len(source['ids'])}")
    expected = {i: (d, m) for i, d, m in zip(source["ids"], source["documents"], source["metadatas"])}
    for record_id, document, metadata in zip(got["ids"], got["documents"], got["metadatas"]):
        if (document, metadata) != expected[record_id]:
            raise RuntimeError(f"record {record_id} differs from the source")
    dimensions = {len(vector) for vector in got["embeddings"]}
    if len(dimensions) != 1:
        raise RuntimeError(f"vectors have different dimensions: {sorted(dimensions)}")
    return dimensions.pop()


def reembed_collection(
    embedding: Embeddings,
    persist_directory: str = CHROMA_PERSIST_DIR,
    source: str = GRANITE_COLLECTION,
    batch_size: int = OPENROUTER_EMBEDDING_BATCH_SIZE,
    dry_run: bool = False,
) -> dict:
    """Build the configured model's collection from the summaries in ``source``.

    Args:
        embedding: The configured embedding model (``load_embedding_model()``).
        persist_directory: Folder of the Chroma store.
        source: The collection whose records are copied.
        batch_size: Records embedded and written per step. With OpenRouter one
            step is one request while it is at most
            ``OPENROUTER_EMBEDDING_BATCH_SIZE``.
        dry_run: Only count what would be embedded; no embedding call and no
            write.

    Returns:
        A report: ``source``, ``target``, ``model``, ``records``,
        ``embedded`` (records embedded by this run), ``batches``,
        ``estimated_tokens`` (characters / 4, of the records still to embed),
        ``figures`` (figure references carried over), ``dimension`` and
        ``status`` (``dry-run``, ``exists`` or ``created``).

    Raises:
        ValueError: If ``source`` is the configured model's own collection.
        RuntimeError: If ``source`` does not exist, or the copy does not match
            the source after embedding; the partial copy is kept for a re-run.
        CollectionError: If the target or its partial copy holds another
            model's vectors.
    """
    model_id = embedding_model_id()
    target = collection_name(model_id)
    if target == source:
        raise ValueError(f"{source!r} already is the collection of {model_id}; choose another EMBEDDING_BACKEND")
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        original = client.get_collection(source)
    except NotFoundError as exc:
        raise RuntimeError(f"{persist_directory} has no collection {source!r} to copy from") from exc
    records = original.get(include=["documents", "metadatas"])
    report = {
        "source": source,
        "target": target,
        "model": model_id,
        "records": len(records["ids"]),
        "embedded": 0,
        "figures": sum(len(json.loads(m.get("image_paths") or "[]")) for m in records["metadatas"]),
        "dimension": None,
    }

    existing = {c.name for c in client.list_collections()}
    if target in existing:
        finished = client.get_collection(target)
        check_collection(target, finished.metadata, model_id)
        try:
            report["dimension"] = _verify(records, finished)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{target!r} already exists but no longer matches {source!r} ({exc}); "
                "delete it to rebuild it"
            ) from exc
        return {**report, "batches": 0, "estimated_tokens": 0, "status": "exists"}

    partial_name = target + PARTIAL_SUFFIX
    done: set[str] = set()
    if partial_name in existing:
        partial = client.get_collection(partial_name)
        check_collection(partial_name, partial.metadata, model_id)
        done = set(partial.get(include=[])["ids"])
    todo = [i for i, record_id in enumerate(records["ids"]) if record_id not in done]
    report["batches"] = math.ceil(len(todo) / batch_size)
    report["estimated_tokens"] = sum(len(records["documents"][i]) for i in todo) // 4
    if dry_run:
        return {**report, "status": "dry-run"}

    partial = client.get_or_create_collection(partial_name, metadata=collection_metadata())
    for step, start in enumerate(range(0, len(todo), batch_size), start=1):
        batch = todo[start:start + batch_size]
        vectors = embedding.embed_documents([records["documents"][i] for i in batch])
        partial.add(
            ids=[records["ids"][i] for i in batch],
            embeddings=vectors,
            documents=[records["documents"][i] for i in batch],
            metadatas=[records["metadatas"][i] for i in batch],
        )
        report["embedded"] += len(batch)
        logger.info("Batch %d/%d: %d records embedded into %s", step, report["batches"], len(batch), partial_name)

    report["dimension"] = _verify(records, partial)
    partial.modify(name=target)
    logger.info("Verified %d records and renamed %s to %s", report["records"], partial_name, target)
    return {**report, "status": "created"}
