"""Index and health endpoints.

``/`` lists what the API exposes and ``/health`` reports whether the vector
store loads. Both stay unversioned so uptime probes have a fixed URL.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.service import StoreUnavailableError

router = APIRouter(tags=["system"])


@router.get("/", summary="List endpoints", description="Returns the API name, version and its endpoints.")
def root(request: Request) -> dict:
    """Human-readable index of what this API exposes."""
    return {
        "name": "Multimodal RAG API",
        "version": request.app.version,
        "docs": "/docs",
        "endpoints": {
            "POST /v1/retrieve": "Retrieve chunks only (text, tables, images)",
            "POST /v1/query": "Retrieve chunks and generate a full answer; an optional llm object picks the model",
            "POST /v1/query/stream": "SSE stream of the answer with the same request body as /v1/query",
            "GET /v1/providers": "List the providers and models the llm object accepts",
            "GET /v1/documents": "List the indexed documents with their chunk, figure and table counts",
            "GET /v1/documents/{document_id}/chunks": "One page of a document's chunks, in reading order",
            "GET /v1/images/{record_id}/{position}": "Fetch one figure of a stored chunk (stable URL)",
            "GET /health": "Service health / database status",
        },
    }


@router.get(
    "/health",
    summary="Health check",
    description=(
        "Returns 200 when the database answers and 503 when it does not. "
        "The first call may take a few seconds while the store loads."
    ),
    response_description="The service is ready.",
    responses={503: {"description": "The database could not be reached."}},
)
async def health(request: Request):
    """Report whether the database can be reached from this runtime.

    Returns 200 when it is and 503 when it is not, so an uptime probe can
    read the status code alone. The first call may take a few seconds while
    the embedding client is built and the database answers a ``SELECT 1``.
    """
    try:
        await request.app.state.service.ensure_ready()
    except StoreUnavailableError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "vector_store": "unavailable", "detail": str(exc)},
        )
    return {"status": "ok", "vector_store": "loaded"}
