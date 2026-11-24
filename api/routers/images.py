"""Figure endpoint: ``GET /v1/images/{record_id}/{position}`` serves one figure
of a stored chunk from disk.

The URL names a record and a position, never a file: the path is looked up in
the ``images`` table (``api/service.py:RAGService.image``) and must lie inside
``IMAGES_DIR``. Because a record's figures never change, the URL is stable and
can be cached.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.responses import FileResponse

from api.openapi_examples import error_responses

router = APIRouter(tags=["images"])


@router.get(
    "/images/{record_id}/{position}",
    summary="Get a figure",
    description=(
        "Returns one figure of a stored chunk, as listed in its `image_urls` or "
        "in a query's `figures`. The URL is stable, and responses may be cached "
        "for a day."
    ),
    response_class=FileResponse,
    response_description="The image file.",
    responses={200: {"content": {"image/jpeg": {}}}, **error_responses(404, 503)},
)
async def serve_image(
    request: Request,
    record_id: Annotated[uuid.UUID, Path(description="The chunk's `record_id`.")],
    position: Annotated[int, Path(ge=0, description="0-based position of the figure in the chunk.")],
) -> FileResponse:
    """Serve one stored figure by record id and position.

    Raises:
        HTTPException: 404 when the record or the figure does not exist.
    """
    path = await request.app.state.service.image(str(record_id), position)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    # FileResponse sends the file in pieces and sets the content type from
    # its extension, plus ETag and Last-Modified for revalidation.
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})
