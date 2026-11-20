"""Shared pytest configuration.

Skips tests marked ``live`` unless the run selects them with ``-m live``, so
the default suite never needs keys, the network, or the real Chroma store.
Offline tests run with the watsonx (Granite) embedding backend whatever
``EMBEDDING_BACKEND`` says in ``.env``, because the backend decides which
collection the store code opens.
"""

from __future__ import annotations

import pytest

import src.embed
import src.retrieval.search


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every ``live`` test as skipped unless ``-m`` mentions ``live``."""
    if "live" in (config.getoption("markexpr") or ""):
        return
    skip_live = pytest.mark.skip(reason="live test: run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def granite_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the watsonx backend in offline tests; live tests keep the configured one."""
    if "live" in request.keywords:
        return
    monkeypatch.setattr(src.embed, "EMBEDDING_BACKEND", "watsonx")
    monkeypatch.setattr(src.retrieval.search, "EMBEDDING_BACKEND", "watsonx")
