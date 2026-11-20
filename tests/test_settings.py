"""Tests for config/settings.py: the LangSmith tracing switch and the embedding backend choice."""

import importlib

import pytest
from langsmith import utils

import config.settings


@pytest.fixture
def reload_settings(monkeypatch):
    """Reload settings under a test environment; restore everything afterwards."""
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "unset-by-test")  # restored at teardown

    def run(**env: str | None):
        for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
            if env.get(name) is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, env[name])
        # load_dotenv never overrides a variable already set, but a deleted one
        # would be filled from .env, so stop it reading the file.
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        utils.get_env_var.cache_clear()
        return importlib.reload(config.settings)

    yield run
    utils.get_env_var.cache_clear()
    monkeypatch.undo()
    importlib.reload(config.settings)
    utils.get_env_var.cache_clear()


def _langsmith_enabled() -> bool:
    utils.get_env_var.cache_clear()
    return utils.tracing_is_enabled({"enabled": None}) is True


def test_langsmith_tracing_false_beats_legacy_true(reload_settings, monkeypatch):
    monkeypatch.setattr("langsmith.run_helpers.get_current_run_tree", lambda: None)
    settings = reload_settings(LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="true")
    assert settings.LANGSMITH_TRACING is False
    assert not _langsmith_enabled()


def test_legacy_flag_alone_still_enables(reload_settings):
    settings = reload_settings(LANGCHAIN_TRACING_V2="true")
    assert settings.LANGSMITH_TRACING is True


def test_tracing_is_off_when_nothing_is_set(reload_settings):
    settings = reload_settings()
    assert settings.LANGSMITH_TRACING is False


def test_openrouter_is_an_embedding_backend(reload_settings, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "openrouter")
    monkeypatch.delenv("OPENROUTER_EMBEDDING_MODEL", raising=False)

    settings = reload_settings()

    assert settings.EMBEDDING_BACKEND == "openrouter"
    assert settings.OPENROUTER_EMBEDDING_MODEL == "nvidia/nemotron-3-embed-1b:free"
    assert settings.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_a_misspelt_embedding_backend_stops_startup(reload_settings, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "openroutr")

    with pytest.raises(ValueError, match="EMBEDDING_BACKEND='openroutr' is not one of: watsonx, local, openrouter"):
        reload_settings()
