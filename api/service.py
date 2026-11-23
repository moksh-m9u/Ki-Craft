"""Backing service for the FastAPI endpoints.

Wraps the retrieval stack (``src.retrieval``) and the corpus reads
(``src/browse.py``) behind a lazy-loaded, thread-safe service so endpoints
stay thin. It holds the two heavy objects, created once: the watsonx
embedding client and the Postgres engine (a connection pool to Neon). All
blocking work (SQL, embedding calls, LLM calls) runs in a worker thread via
``asyncio.to_thread`` so FastAPI's event loop is never blocked.

Every query, with the default model or a caller-chosen one, goes through
``stream_answer`` and so through the one generation path
(``answer_query_stream``). The LangSmith UI shows one ``AnswerQuery`` run with
``RetrieveChunks`` and ``GenerateAnswerStream`` nested underneath.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import (
    GENERATION_MAX_IMAGES,
    GENERATION_MODEL,
    DATABASE_URL,
    GENERATION_PROVIDER,
    RETRIEVAL_K,
)
from src.browse import document_chunks, figure_path, list_documents
from src.db.connection import create_db_engine, ping
from src.documents import chunk_text_and_tables, figure_file, stored_image_paths
from src.embed import load_embedding_model
from src.llm import accepts_images, default_llm
from src.logger import get_logger
from src.retrieval.citations import find_citations
from src.retrieval.generate import StreamEvent, answer_query_stream
from src.retrieval.prompt import PromptFigure
from src.retrieval.search import PostgresRetriever, retrieve_chunks

from api.schemas import (
    ChunkPage,
    ChunkPayload,
    CitationsPayload,
    DocumentList,
    DocumentSummaryPayload,
    FigurePayload,
    RetrieveResponse,
    StoredChunk,
)

logger = get_logger(__name__)

# Metadata keys that are excluded from the API payload's ``metadata`` dict to
# keep responses small (the full content is exposed via dedicated fields).
_EXCLUDED_META_KEYS = {"raw_text", "tables_html", "image_paths"}


def _image_urls(chunk: Document) -> list[str]:
    """The stable URL of each of a chunk's figures; none without a record id."""
    if not chunk.id:
        return []
    return [f"/v1/images/{chunk.id}/{position}" for position in range(len(stored_image_paths(chunk.metadata or {})))]


class StoreUnavailableError(RuntimeError):
    """The embedding client or the database could not be loaded or reached.

    ``api/main.py`` maps it to HTTP 503: the service cannot answer right now,
    and ``/health`` reports the same.
    """


class RAGService:
    """Lazily connects to the database and exposes retrieve, answer and browse operations."""

    def __init__(self) -> None:
        """Create the service with nothing loaded; ``ensure_ready`` connects."""
        self._engine = None
        self._embedding = None
        self._load_lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        """Connect on first use (idempotent, thread-safe).

        Raises:
            StoreUnavailableError: If the embedding client cannot be built or
                the database cannot be reached. The next call tries again.
        """
        if self._engine is None:
            async with self._load_lock:
                if self._engine is None:
                    try:
                        await asyncio.to_thread(self._load)
                    except Exception as exc:
                        raise StoreUnavailableError(f"Database unavailable: {exc}") from exc

    def _load(self) -> None:
        """Build the embedding client and the engine, and check the database answers."""
        logger.info("Loading the embedding model and connecting to Postgres...")
        embedding = load_embedding_model()
        engine = create_db_engine(DATABASE_URL)
        ping(engine)
        self._embedding, self._engine = embedding, engine
        logger.info("Postgres ready.")

    def get_retriever(self, top_k: int | None = None) -> PostgresRetriever:
        """Return a retriever for ``top_k`` chunks (``RETRIEVAL_K`` when not given).

        A retriever only holds references to the engine and the embedding
        client, so building one per request costs nothing.
        """
        if self._engine is None:
            raise RuntimeError("The database is not connected")
        return PostgresRetriever(self._engine, self._embedding, k=top_k or RETRIEVAL_K)

    async def retrieve(self, query: str, top_k: int) -> RetrieveResponse:
        """Retrieve chunks for a query without generating an answer."""
        await self.ensure_ready()
        retriever = self.get_retriever(top_k)
        chunks = await asyncio.to_thread(retrieve_chunks, retriever, query)
        return self.to_retrieve_response(query, chunks)

    def stream_answer(
        self,
        query: str,
        top_k: int,
        max_images: int = GENERATION_MAX_IMAGES,
        llm: BaseChatModel | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """Retrieve chunks and return the answer as a stream of events.

        Blocking: call it from a worker thread, after ``ensure_ready``. This is
        the one place that picks the default model when no ``llm`` is given.

        Args:
            query: The user's question.
            top_k: How many chunks to retrieve.
            max_images: Most images to send to the model (0 = all).
            llm: A chat model from ``src/llm.py:create_llm``; the default
                Gemini model when omitted.
            provider_id: The catalogue provider of ``llm``, used to price
                the answer. Ignored when ``llm`` is omitted.
            model_id: The catalogue id of ``llm``, recorded in usage. Ignored
                when ``llm`` is omitted. With ``provider_id`` it decides
                whether figures are sent: a text-only model gets none.

        Returns:
            The generator from ``answer_query_stream``: ``("chunks", ...)``,
            ``("figures", ...)``, then ``("token", ...)`` per answer piece,
            then ``("usage", ...)``.
            Nothing runs until it is iterated.
        """
        if llm is None:
            llm, provider_id, model_id = default_llm(), GENERATION_PROVIDER, GENERATION_MODEL
        return answer_query_stream(
            self.get_retriever(top_k),
            query,
            llm,
            provider_id,
            model_id,
            max_images=max_images,
            send_images=accepts_images(provider_id, model_id),
        )

    async def answer(
        self,
        query: str,
        top_k: int,
        max_images: int = GENERATION_MAX_IMAGES,
        llm: BaseChatModel | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
    ) -> tuple[str, RetrieveResponse, list[FigurePayload], dict]:
        """Retrieve chunks and generate a full answer.

        Consumes every event from :meth:`stream_answer`, so the arguments are
        the same and retrieval and generation stay under a single trace.

        Returns:
            A tuple of ``(answer, retrieve_response, figures, usage)`` where
            ``figures`` are the figures the model saw and ``usage`` carries
            token counts, model, latency and cost estimate.
        """
        await self.ensure_ready()

        def _run() -> tuple[str, list, list, dict]:
            """Consume the event stream and collect each kind of event."""
            chunks: list = []
            figures: list = []
            tokens: list[str] = []
            usage: dict = {}
            for kind, value in self.stream_answer(query, top_k, max_images, llm, provider_id, model_id):
                if kind == "chunks":
                    chunks = value
                elif kind == "figures":
                    figures = value
                elif kind == "token":
                    tokens.append(value)
                else:
                    usage = value
            return "".join(tokens), chunks, figures, usage

        answer, chunks, figures, usage = await asyncio.to_thread(_run)
        return answer, self.to_retrieve_response(query, chunks), self.to_figure_payloads(figures), usage

    async def image(self, record_id: str, position: int) -> Path | None:
        """Find one figure of a stored record, for ``/v1/images/{record_id}/{position}``.

        The file path is read from the ``images`` table, never taken from the
        caller, and must lie inside ``IMAGES_DIR``.

        Args:
            record_id: The chunk's record id (a UUID string).
            position: 0-based index into the record's figures.

        Returns:
            The figure's file, or None when the record, the position or the
            file does not exist.

        Raises:
            StoreUnavailableError: If the store cannot be loaded.
        """
        await self.ensure_ready()
        stored = await asyncio.to_thread(figure_path, self._engine, record_id, position)
        return figure_file(stored) if stored else None

    async def documents(self) -> DocumentList:
        """List every indexed document with its chunk, figure and table counts.

        Raises:
            StoreUnavailableError: If the store cannot be loaded.
        """
        await self.ensure_ready()
        summaries = await asyncio.to_thread(list_documents, self._engine)
        items = [DocumentSummaryPayload(**vars(summary)) for summary in summaries]
        return DocumentList(items=items, total=len(items))

    async def document_chunks(self, document_id: str, offset: int, limit: int) -> ChunkPage | None:
        """Return one page of a document's chunks, in reading order.

        Args:
            document_id: The document's name, such as ``lm317``.
            offset: Chunks to skip from the start of the document.
            limit: Most chunks to return.

        Returns:
            The page, or None when no document has that name. An offset past
            the end gives an empty page, not None.

        Raises:
            StoreUnavailableError: If the store cannot be loaded.
        """
        await self.ensure_ready()
        chunks, total = await asyncio.to_thread(document_chunks, self._engine, document_id, offset, limit)
        if total == 0:
            return None
        items = []
        for chunk in chunks:
            tables = chunk_text_and_tables(chunk)["tables_html"]
            image_urls = _image_urls(chunk)
            items.append(
                StoredChunk(
                    record_id=chunk.id,
                    chunk_id=chunk.metadata.get("chunk_id") or 0,
                    source=document_id,
                    enhanced_content=chunk.page_content,
                    raw_text=chunk.metadata.get("raw_text") or "",
                    tables_html=tables,
                    image_urls=image_urls,
                    has_table=bool(tables),
                    has_image=bool(image_urls),
                )
            )
        return ChunkPage(items=items, total=total, offset=offset, limit=limit)

    @staticmethod
    def to_figure_payloads(figures: list[PromptFigure]) -> list[FigurePayload]:
        """Describe the figures sent to the model by label, URL, chunk and source.

        A figure's ``chunk_id`` is its chunk's 1-based position, the same
        number the response's ``chunks`` use.
        """
        return [
            FigurePayload(
                label=figure.label,
                url=f"/v1/images/{figure.record_id}/{figure.position}",
                chunk_id=figure.chunk_number,
                source=figure.source,
            )
            for figure in figures
            if figure.record_id
        ]

    @staticmethod
    def to_citations(answer: str, chunk_count: int, figures: list[FigurePayload]) -> CitationsPayload:
        """Check the answer's ``[C1]`` and ``[IMG1]`` citations against the prompt.

        Args:
            answer: The finished answer text.
            chunk_count: Chunks in the response; ``C1`` to ``C<chunk_count>`` exist.
            figures: The figures listed in the response; only their labels
                count as sent.

        Returns:
            The cited chunks and figures, and the cited labels that match neither.
        """
        found = find_citations(answer, chunk_count, [figure.label for figure in figures])
        return CitationsPayload(chunks=found.chunks, figures=found.figures, unknown=found.unknown)

    def to_retrieve_response(self, query: str, chunks: list) -> RetrieveResponse:
        """Build the API response for a set of retrieved chunks.

        Each figure stored as a file gets a stable URL,
        ``/v1/images/{record_id}/{position}``, where ``record_id`` is the
        chunk's record id and ``position`` counts from 0 within the
        chunk. No image file is read: the response carries URLs, not bytes,
        so ``image_count`` and ``total_images`` count those URLs.
        """
        payloads: list[ChunkPayload] = []
        total_images = 0
        total_tables = 0
        for i, chunk in enumerate(chunks):
            raw = chunk_text_and_tables(chunk)
            tables = raw["tables_html"]
            image_urls = _image_urls(chunk)
            metadata = {
                k: v
                for k, v in (chunk.metadata or {}).items()
                if k not in _EXCLUDED_META_KEYS
            }
            payloads.append(
                ChunkPayload(
                    chunk_id=i + 1,
                    enhanced_content=chunk.page_content or "",
                    raw_text=raw["raw_text"],
                    tables_html=tables,
                    image_count=len(image_urls),
                    record_id=chunk.id,
                    image_urls=image_urls,
                    has_table=bool(tables),
                    has_image=bool(image_urls),
                    source=metadata.get("source"),
                    metadata=metadata,
                )
            )
            total_images += len(image_urls)
            total_tables += len(tables)

        return RetrieveResponse(
            query=query,
            num_chunks=len(chunks),
            total_images=total_images,
            total_tables=total_tables,
            chunks=payloads,
        )
