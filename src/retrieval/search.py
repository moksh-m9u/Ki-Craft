"""Retrieval: find the stored chunks that answer a question.

The chunks live in Postgres (``src/db/schema.py``); pgvector compares the
question's embedding with every chunk's. ``PostgresRetriever`` is what the API
builds per request, ``search_postgres`` applies the search type (similarity,
MMR, or score threshold, with parameters from ``config/settings.py``), and
``retrieve_chunks`` is a traced run so retrieval shows up as its own child
run inside the ``AnswerQuery`` trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langsmith import traceable
from sqlalchemy import Engine, Row

from config.settings import (
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RETRIEVAL_LAMBDA_MULT,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_SEARCH_TYPE,
)
from src.db.schema import chunks
from src.documents import chunk_text_and_tables, document_from_row, stored_chunks_query, stored_image_count
from src.logger import get_logger

logger = get_logger(__name__)


def nearest_chunks(engine: Engine, vector: list[float], n: int) -> list[Row]:
    """The ``n`` chunks whose embeddings are closest to ``vector``, nearest first.

    pgvector's ``<=>`` operator is cosine distance (0 = same direction), the
    metric the Chroma collection used. Every row is compared: an exact scan,
    fast at this size; an HNSW index can be added when the corpus grows.

    Args:
        engine: The Postgres engine (``src/db/connection.py``).
        vector: The question's embedding.
        n: How many chunks to return.

    Returns:
        Rows of ``stored_chunks_query`` plus ``embedding`` and ``distance``.
    """
    distance = chunks.c.embedding.cosine_distance(vector)
    # Ties (identical embeddings) are broken by record id so the order never varies.
    query = (
        stored_chunks_query(chunks.c.embedding, distance.label("distance"))
        .order_by(distance, chunks.c.id)
        .limit(n)
    )
    with engine.connect() as connection:
        return list(connection.execute(query))


def search_postgres(engine: Engine, vector: list[float], search_type: str, k: int) -> list[Document]:
    """Find chunks for one question embedding with the configured search type.

    Each type does what LangChain's Chroma retriever did, so answers do not
    change with the store:

    * ``similarity``: the ``k`` nearest chunks.
    * ``mmr``: the ``RETRIEVAL_FETCH_K`` nearest (at least ``k``), then
      LangChain's ``maximal_marginal_relevance`` keeps ``k`` that are
      relevant and unlike each other (``RETRIEVAL_LAMBDA_MULT``), returned
      nearest first.
    * ``similarity_score_threshold``: the ``k`` nearest whose score,
      1 minus the cosine distance, is at least ``RETRIEVAL_SCORE_THRESHOLD``.

    Args:
        engine: The Postgres engine.
        vector: The question's embedding.
        search_type: One of ``RETRIEVAL_SEARCH_TYPES``.
        k: How many chunks to return (fewer for the threshold type).

    Returns:
        The chunks as ``Document`` objects, best first.

    Raises:
        ValueError: If ``search_type`` is not a known search type.
    """
    if search_type == "similarity":
        rows = nearest_chunks(engine, vector, k)
    elif search_type == "mmr":
        pool = nearest_chunks(engine, vector, max(k, RETRIEVAL_FETCH_K))
        picked = maximal_marginal_relevance(
            np.array(vector, dtype=np.float32), [row.embedding for row in pool], RETRIEVAL_LAMBDA_MULT, k
        )
        # Nearest first, as Chroma returned them: the order sets the C1..C5
        # labels and which figures fit under the image cap.
        rows = [pool[i] for i in sorted(picked)]
    elif search_type == "similarity_score_threshold":
        rows = [row for row in nearest_chunks(engine, vector, k) if 1 - row.distance >= RETRIEVAL_SCORE_THRESHOLD]
    else:
        raise ValueError(f"Unknown search type {search_type!r}")
    return [document_from_row(row) for row in rows]


@dataclass(frozen=True)
class PostgresRetriever:
    """Embeds a question and searches Postgres; ``invoke`` matches a LangChain retriever's.

    Attributes:
        engine: The Postgres engine.
        embedding_model: The model the chunks were embedded with.
        k: How many chunks each question returns.
        search_type: One of ``RETRIEVAL_SEARCH_TYPES``.
    """

    engine: Engine
    embedding_model: Embeddings
    k: int = RETRIEVAL_K
    search_type: str = RETRIEVAL_SEARCH_TYPE

    def invoke(self, query: str) -> list[Document]:
        """Return the chunks for ``query`` (one watsonx call, one or two SQL queries)."""
        return search_postgres(self.engine, self.embedding_model.embed_query(query), self.search_type, self.k)


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
