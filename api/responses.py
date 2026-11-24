"""Turn the backend's results into the API's response bodies.

The service (``api/service.py``) runs retrieval, generation and the corpus
reads and gets back project objects: :class:`src.chunk.Chunk`,
:class:`src.retrieval.prompt.PromptFigure` and
:class:`src.browse.DocumentSummary`. The functions here map those onto the
Pydantic models in ``api/schemas.py``, which is the only place figure URLs
are built. Nothing here touches the database, a model or a file.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.browse import DocumentSummary
from src.chunk import Chunk
from src.retrieval.citations import find_citations
from src.retrieval.prompt import PromptFigure

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


def image_url(record_id: str, position: int) -> str:
    """The stable URL of one figure: ``/v1/images/{record_id}/{position}``."""
    return f"/v1/images/{record_id}/{position}"


def _image_urls(chunk: Chunk) -> list[str]:
    """The URL of each of a chunk's figures; none without a record id."""
    if not chunk.record_id:
        return []
    return [image_url(chunk.record_id, position) for position in range(len(chunk.image_paths))]


def retrieve_response(query: str, chunks: Sequence[Chunk]) -> RetrieveResponse:
    """Build the response for a set of retrieved chunks.

    Each chunk's ``chunk_id`` in the response is its 1-based position in the
    list, the number the prompt's ``C1``, ``C2``, ... labels use. No image
    file is read: the response carries URLs, not bytes, so ``image_count``
    and ``total_images`` count those URLs.

    Args:
        query: The question the chunks were retrieved for.
        chunks: The retrieved chunks, best first.

    Returns:
        The response body for ``/v1/retrieve`` and the start of ``/v1/query``.
    """
    payloads: list[ChunkPayload] = []
    for position, chunk in enumerate(chunks, start=1):
        image_urls = _image_urls(chunk)
        payloads.append(
            ChunkPayload(
                chunk_id=position,
                enhanced_content=chunk.summary or "",
                raw_text=chunk.raw_text,
                tables_html=chunk.tables_html,
                image_count=len(image_urls),
                record_id=chunk.record_id,
                image_urls=image_urls,
                has_table=chunk.has_table,
                has_image=bool(image_urls),
                source=chunk.source,
                metadata={
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "has_table": chunk.has_table,
                    "has_image": chunk.has_image,
                },
            )
        )
    return RetrieveResponse(
        query=query,
        num_chunks=len(payloads),
        total_images=sum(payload.image_count for payload in payloads),
        total_tables=sum(len(payload.tables_html) for payload in payloads),
        chunks=payloads,
    )


def figure_payloads(figures: Sequence[PromptFigure]) -> list[FigurePayload]:
    """Describe the figures sent to the model by label, URL, chunk and source.

    A figure's ``chunk_id`` is its chunk's 1-based position, the same number
    the response's ``chunks`` use. A figure without a record id has no URL
    and is left out.
    """
    return [
        FigurePayload(
            label=figure.label,
            url=image_url(figure.record_id, figure.position),
            chunk_id=figure.chunk_number,
            source=figure.source,
        )
        for figure in figures
        if figure.record_id
    ]


def citations_payload(answer: str, chunk_count: int, figures: Sequence[FigurePayload]) -> CitationsPayload:
    """Check the answer's ``[C1]`` and ``[IMG1]`` citations against the prompt.

    Args:
        answer: The finished answer text.
        chunk_count: Chunks in the response; ``C1`` to ``C<chunk_count>`` exist.
        figures: The figures listed in the response; only their labels count
            as sent.

    Returns:
        The cited chunks and figures, and the cited labels that match neither.
    """
    found = find_citations(answer, chunk_count, [figure.label for figure in figures])
    return CitationsPayload(chunks=found.chunks, figures=found.figures, unknown=found.unknown)


def document_list(summaries: Sequence[DocumentSummary]) -> DocumentList:
    """Build the ``/v1/documents`` body from the stored documents' counts."""
    items = [DocumentSummaryPayload(**vars(summary)) for summary in summaries]
    return DocumentList(items=items, total=len(items))


def chunk_page(chunks: Sequence[Chunk], total: int, offset: int, limit: int) -> ChunkPage:
    """Build one page of ``/v1/documents/{document_id}/chunks``.

    Args:
        chunks: The chunks on this page, in reading order.
        total: Chunks in the whole document.
        offset: Chunks skipped before this page.
        limit: Most chunks a page holds, as requested.

    Returns:
        The page body; each chunk keeps its stored ``chunk_id``.
    """
    items = []
    for chunk in chunks:
        image_urls = _image_urls(chunk)
        items.append(
            StoredChunk(
                record_id=chunk.record_id,
                chunk_id=chunk.chunk_id or 0,
                source=chunk.source,
                enhanced_content=chunk.summary,
                raw_text=chunk.raw_text,
                tables_html=chunk.tables_html,
                image_urls=image_urls,
                has_table=chunk.has_table,
                has_image=bool(image_urls),
            )
        )
    return ChunkPage(items=items, total=total, offset=offset, limit=limit)
