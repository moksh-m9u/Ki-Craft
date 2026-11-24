"""Browsing endpoints: which documents are indexed, and each one's chunks.

``GET /v1/documents`` lists the documents; ``GET /v1/documents/{document_id}/chunks``
returns a document's chunks in reading order, one page at a time. The
dashboard's Chunk Explorer, Compare and Analytics pages read these instead of
the ``json/`` exports. No embedding or vector search runs: the service reads
stored records (``src/browse.py``).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Request

from api.openapi_examples import error_responses
from api.responses import chunk_page, document_list
from api.schemas import ChunkPage, DocumentList

router = APIRouter(tags=["documents"])

# A page is capped so one request stays small. 200 is more than the largest
# stored document (77 chunks), so a client can still read one in a single call.
_MAX_PAGE_SIZE = 200


@router.get(
    "/documents",
    response_model=DocumentList,
    summary="List documents",
    description="Returns every indexed document with its chunk, figure and table counts, sorted by `id`.",
    response_description="The indexed documents.",
    responses=error_responses(503),
)
async def list_documents(request: Request) -> DocumentList:
    """List the indexed documents."""
    return document_list(await request.app.state.service.documents())


@router.get(
    "/documents/{document_id}/chunks",
    response_model=ChunkPage,
    summary="List a document's chunks",
    description=(
        "Returns a document's chunks in reading order: summary, original text, "
        "tables as HTML and figure URLs. Page with `offset` and `limit`; `total` "
        "is the document's chunk count. An `offset` past the end returns an empty page."
    ),
    response_description="One page of chunks.",
    responses=error_responses(404, 503),
)
async def list_document_chunks(
    request: Request,
    document_id: Annotated[str, Path(description="The document's `id`, as `GET /v1/documents` lists it.")],
    offset: Annotated[int, Query(ge=0, description="Chunks to skip from the start of the document.")] = 0,
    limit: Annotated[int, Query(ge=1, le=_MAX_PAGE_SIZE, description="Most chunks to return.")] = 50,
) -> ChunkPage:
    """Return one page of a document's chunks.

    Raises:
        HTTPException: 404 when no document has that id.
    """
    chunks, total = await request.app.state.service.document_chunks(document_id, offset, limit)
    if total == 0:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    return chunk_page(chunks, total, offset, limit)
