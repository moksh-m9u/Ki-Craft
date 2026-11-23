"""Provider endpoint: ``GET /v1/providers`` lists the providers and models that a
query's ``llm`` object accepts.

The list is ``config/providers.py``, the same catalogue ``src/llm.py`` checks a
request against, so a client picks from exactly what the server will build.
Server internals (each provider's protocol and base URL) are left out. No
store or model is touched, so it answers even when the database is down.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import ModelPayload, ProviderList, ProviderPayload
from config.providers import ALL_PROVIDERS

router = APIRouter(tags=["providers"])


@router.get(
    "/providers",
    response_model=ProviderList,
    summary="List providers and models",
    description=(
        "Returns every provider and model a query's `llm` object accepts, with "
        "whether each model reads images and takes a temperature, its default "
        "output budget and its prices. Any other provider or model is a 422."
    ),
    response_description="The catalogue of providers and models.",
)
def list_providers() -> ProviderList:
    """List the providers and models in the catalogue."""
    items = [
        ProviderPayload(
            id=provider.id,
            display_name=provider.display_name,
            models=[
                ModelPayload(
                    id=model.id,
                    display_name=model.display_name,
                    supports_vision=model.supports_vision,
                    accepts_temperature=model.accepts_temperature,
                    default_max_tokens=model.default_max_tokens,
                    input_price_per_1m=model.input_price_per_1m,
                    output_price_per_1m=model.output_price_per_1m,
                )
                for model in provider.models
            ],
        )
        for provider in ALL_PROVIDERS
    ]
    return ProviderList(items=items, total=len(items))
