"""Retrieve-only endpoint: ``POST /v1/retrieve`` returns the chunks for a
question, with no generated answer.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Request

from api.openapi_examples import RETRIEVE_BODY_EXAMPLES, error_responses
from api.responses import retrieve_response
from api.schemas import RetrieveRequest, RetrieveResponse

router = APIRouter(tags=["retrieval"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve chunks",
    description=(
        "Returns the `top_k` chunks most relevant to the question, with their "
        "text, tables and figure URLs. No answer is generated and no model "
        "provider is called."
    ),
    response_description="The retrieved chunks, most relevant first.",
    responses=error_responses(503),
)
async def retrieve(
    request: Request,
    req: Annotated[RetrieveRequest, Body(openapi_examples=RETRIEVE_BODY_EXAMPLES)],
) -> RetrieveResponse:
    """Retrieve chunks for a query; returns text, tables and images."""
    chunks = await request.app.state.service.retrieve(req.query, req.top_k)
    return retrieve_response(req.query, chunks)
