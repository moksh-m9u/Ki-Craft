"""Tests for the search type setting in ``config/settings.py``.

The Postgres search itself is tested in ``tests/test_search_postgres.py``."""

from __future__ import annotations

import pytest

import config.settings as settings


def test_env_choice_reads_an_allowed_value(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_SEARCH_TYPE", "similarity")
    assert settings._env_choice("RETRIEVAL_SEARCH_TYPE", "mmr", ("similarity", "mmr")) == "similarity"


def test_env_choice_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_SEARCH_TYPE", raising=False)
    assert settings._env_choice("RETRIEVAL_SEARCH_TYPE", "mmr", ("similarity", "mmr")) == "mmr"


def test_env_choice_names_the_allowed_values(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_SEARCH_TYPE", "MMR")
    with pytest.raises(ValueError, match="'MMR' is not one of: similarity, mmr"):
        settings._env_choice("RETRIEVAL_SEARCH_TYPE", "mmr", ("similarity", "mmr"))
