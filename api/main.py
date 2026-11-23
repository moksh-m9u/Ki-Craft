"""Multimodal RAG HTTP API: the FastAPI application.

This module only assembles the app: it creates it, adds CORS, sets up the
lifespan that holds one ``RAGService``, turns errors into HTTP status codes,
and includes the routers in ``api/routers/``, where the endpoints live. The
routers declare short paths (``/query``); the ``/v1`` prefix is added here.

Run:
    uvicorn api.main:app --reload --port 8000

Interactive docs live at ``http://localhost:8000/docs``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import documents, images, providers, query, retrieval, system
from api.service import RAGService, StoreUnavailableError
from config.settings import CORS_ALLOWED_ORIGINS
from src.llm import LLMProviderError
from src.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hold one RAG service (loaded lazily on first request)."""
    app.state.service = RAGService()
    yield


_DESCRIPTION = """
Answers questions about electronics documents (datasheets and application
notes) from their text, tables and figures.

A query retrieves the most relevant chunks of the indexed documents, then a
model writes an answer from them. Each chunk carries its text, its tables as
HTML and the URLs of its figures; figures are served by `GET /v1/images/...`.

**Models.** Without an `llm` object the server's default model answers. With
one, the answer comes from the provider and model you name, using your API key.

**Errors.** Error responses have the body `{"detail": "..."}`. A failing model
provider returns 502; an unreachable database returns 503.
"""

# Tag order and descriptions for the /docs sidebar.
_TAGS = [
    {"name": "query", "description": "Retrieve chunks and generate an answer."},
    {"name": "retrieval", "description": "Retrieve chunks without generating an answer."},
    {"name": "providers", "description": "The providers and models a query can use."},
    {"name": "documents", "description": "Browse the indexed documents and their chunks."},
    {"name": "images", "description": "Figures of stored chunks."},
    {"name": "system", "description": "Index and health check."},
]

app = FastAPI(
    title="Multimodal RAG API",
    version="0.2.0",
    description=_DESCRIPTION,
    openapi_tags=_TAGS,
    lifespan=lifespan,
)

# Browser access: allow the configured origins (e.g. the local Vite dev server,
# and later the production frontend URL).  Non-browser callers are unaffected.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """A body that fails validation: 422, without echoing the input.

    FastAPI's own handler returns each Pydantic error with an ``input`` field
    holding the value that failed. For a missing ``llm.model`` that value is
    the whole ``llm`` object, API key included, so ``input`` is dropped and
    ``type``, ``loc`` and ``msg`` (and ``ctx`` when present) are kept.
    """
    errors = [{k: v for k, v in error.items() if k != "input"} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


@app.exception_handler(LLMProviderError)
async def llm_provider_error(request: Request, exc: LLMProviderError) -> JSONResponse:
    """The model provider failed or wrote nothing: 502 Bad Gateway."""
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(StoreUnavailableError)
async def store_unavailable(request: Request, exc: StoreUnavailableError) -> JSONResponse:
    """The database could not be reached: 503 Service Unavailable."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Any other failure: 500, with the details in the log, not the response."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# "/" and "/health" stay unversioned so uptime probes keep a fixed URL; every
# other endpoint is under /v1, so a later /v2 can change them without breaking
# existing clients.
app.include_router(system.router)
app.include_router(retrieval.router, prefix="/v1")
app.include_router(query.router, prefix="/v1")
app.include_router(providers.router, prefix="/v1")
app.include_router(documents.router, prefix="/v1")
app.include_router(images.router, prefix="/v1")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
