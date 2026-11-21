"""Vector store loading and document retrieval.

Loads the configured embedding model's collection from the persisted Chroma
store and builds the retriever used by the answer generation layer. The store
holds one collection per embedding model (``src/embed.py:collection_name``),
and :func:`load_vector_store` refuses a collection made by another model,
because its distances to the query would mean nothing. The search type
(similarity, MMR, or score threshold) and its parameters come from
``config/settings.py``. ``retrieve_chunks`` is a traced run so retrieval shows
up as its own child run inside the ``AnswerQuery`` trace.
"""

from __future__ import annotations

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langsmith import traceable

from config.settings import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RETRIEVAL_LAMBDA_MULT,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_SEARCH_TYPE,
)
from src.documents import chunk_text_and_tables, stored_image_count
from src.embed import GRANITE_COLLECTION, collection_metadata, collection_name, embedding_model_id, vector_space
from src.logger import get_logger

logger = get_logger(__name__)


class CollectionError(RuntimeError):
    """The store has no usable collection for the configured embedding model."""


def check_collection(name: str, metadata: dict | None, model_id: str) -> None:
    """Raise :class:`CollectionError` unless ``model_id`` made the collection's vectors.

    A collection records its model as ``embedding_model``. The original
    Granite collection (``langchain``) predates that key, so a ``langchain``
    collection without it counts as Granite's.

    Args:
        name: The collection's name.
        metadata: The collection's metadata.
        model_id: The configured embedding model.

    Raises:
        CollectionError: If the collection was made by another model.
    """
    stored = (metadata or {}).get("embedding_model")
    if stored is None and name == GRANITE_COLLECTION:
        stored = EMBEDDING_MODEL
    if stored != vector_space(model_id):
        raise CollectionError(
            f"Embedding configuration mismatch in collection {name!r}: it was made by "
            f"{stored or 'an unrecorded model'}, but the configured model is {model_id} "
            f"(EMBEDDING_BACKEND={EMBEDDING_BACKEND}). Query it with its own model, or build "
            "a collection for this one with: python -m scripts.reembed"
        )


def load_vector_store(
    embedding_model: Embeddings,
    persist_directory: str = CHROMA_PERSIST_DIR,
    create: bool = False,
) -> Chroma:
    """Load the configured embedding model's collection from the Chroma store.

    The API reads through it and ingestion writes through it (``create=True``),
    so both open the same collection the same way.

    Args:
        embedding_model: The embedding model to query with; it must be the
            configured one, which made the collection's vectors.
        persist_directory: Folder of the Chroma store.
        create: Create the collection if it is missing (ingestion). Serving
            never creates one: a new, empty collection would answer every
            question with zero chunks instead of an error.

    Returns:
        The Chroma store bound to the model's collection.

    Raises:
        CollectionError: If the collection is missing or empty (and
            ``create`` is false), or was made by another model.
    """
    model_id = embedding_model_id()
    name = collection_name(model_id)
    client = chromadb.PersistentClient(path=persist_directory)
    if create:
        collection = client.get_or_create_collection(name, metadata=collection_metadata())
    else:
        try:
            collection = client.get_collection(name)
        except NotFoundError as exc:
            raise CollectionError(
                f"{persist_directory} has no collection for embedding model {model_id} "
                f"(EMBEDDING_BACKEND={EMBEDDING_BACKEND}). Build it from the stored summaries "
                "with: python -m scripts.reembed"
            ) from exc
        if collection.count() == 0:
            raise CollectionError(
                f"Collection {name!r} in {persist_directory} holds no records. Delete it, then rebuild it "
                "with: python -m scripts.reembed (for Granite: python -m scripts.ingest json/<name>.json)"
            )
    check_collection(name, collection.metadata, model_id)
    logger.info("Loading collection %s (%s, %d records) from %s", name, model_id, collection.count(), persist_directory)
    return Chroma(client=client, collection_name=name, embedding_function=embedding_model)


def search_kwargs(search_type: str, k: int) -> dict:
    """Build the ``search_kwargs`` LangChain needs for one search type.

    Each search type takes different parameters, and Chroma rejects ones that
    do not belong to it, so only the matching ones are included.

    Args:
        search_type: One of ``RETRIEVAL_SEARCH_TYPES`` in ``config/settings.py``.
        k: How many chunks to return.

    Returns:
        The keyword arguments for ``as_retriever(search_kwargs=...)``.

    Raises:
        ValueError: If ``search_type`` is not a known search type.
    """
    if search_type == "similarity":
        return {"k": k}
    if search_type == "mmr":
        # The pool must be at least k, or MMR has fewer candidates than it returns.
        return {"k": k, "fetch_k": max(k, RETRIEVAL_FETCH_K), "lambda_mult": RETRIEVAL_LAMBDA_MULT}
    if search_type == "similarity_score_threshold":
        return {"k": k, "score_threshold": RETRIEVAL_SCORE_THRESHOLD}
    raise ValueError(f"Unknown search type {search_type!r}")


def build_retriever(db: Chroma, k: int = RETRIEVAL_K):
    """Build a retriever with the configured search type.

    Args:
        db: The loaded Chroma vector store.
        k: How many chunks each query returns.

    Returns:
        A LangChain retriever whose ``invoke(question)`` returns ``k`` chunks
        (fewer for ``similarity_score_threshold``).
    """
    kwargs = search_kwargs(RETRIEVAL_SEARCH_TYPE, k)
    logger.debug("Building retriever (search_type=%s, %s)", RETRIEVAL_SEARCH_TYPE, kwargs)
    return db.as_retriever(search_type=RETRIEVAL_SEARCH_TYPE, search_kwargs=kwargs)


@traceable(run_type="retriever", name="RetrieveChunks")
def retrieve_chunks(retriever, query: str) -> list[Document]:
    """Run retrieval and return matching chunks.

    Args:
        retriever: The configured retriever.
        query: The user's question.

    Returns:
        List of retrieved document chunks.
    """
    chunks = retriever.invoke(query)
    total_images = sum(stored_image_count(chunk) for chunk in chunks)
    total_tables = sum(len(chunk_text_and_tables(chunk)["tables_html"]) for chunk in chunks)
    logger.info(
        "Retrieved %d chunks (%d images, %d tables)",
        len(chunks),
        total_images,
        total_tables,
    )
    return chunks
