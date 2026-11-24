"""Tests for src/retrieval/usage.py: adding up streamed token counts and pricing them."""

from src.retrieval.usage import capture_usage, estimate_cost


class _FakeMsg:
    def __init__(self, usage_metadata=None):
        self.usage_metadata = usage_metadata
        self.content = ""


def test_capture_usage_adds_each_chunk():
    """Gemini chunks carry the tokens each chunk added; the totals are their sum."""
    usage = {}
    capture_usage(_FakeMsg({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}), usage)
    capture_usage(_FakeMsg({"input_tokens": 0, "output_tokens": 3, "total_tokens": 3}), usage)
    assert usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_capture_usage_single_final_chunk():
    """OpenAI-style streams send the whole count once, on the last chunk."""
    usage = {}
    capture_usage(_FakeMsg(), usage)
    capture_usage(_FakeMsg({"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}), usage)
    assert usage == {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}


def test_capture_usage_ignores_non_int():
    usage = {}
    capture_usage(_FakeMsg({"input_tokens": "nope", "output_tokens": 1, "total_tokens": 10}), usage)
    assert usage == {"output_tokens": 1, "total_tokens": 10}


def test_capture_usage_no_metadata():
    usage = {}
    capture_usage(_FakeMsg(), usage)
    assert usage == {}


def test_estimate_cost_uses_the_models_own_prices():
    """GPT-4o is priced as GPT-4o (2.50 in, 10.00 out per 1M), not as the default model."""
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    assert estimate_cost("openai", "gpt-4o", usage) == 3.5
    assert estimate_cost("gemini", "gemini-2.5-flash-lite", usage) == 0.14


def test_estimate_cost_unknown_model_is_none():
    assert estimate_cost("openai", "not-a-model", {"input_tokens": 10}) is None
