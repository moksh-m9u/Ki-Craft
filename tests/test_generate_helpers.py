"""Tests for the pure helpers in src/retrieval/generate.py and prompt.py.

Cloud/model dependencies are optional; the module is skipped entirely when
they are not installed so the suite stays lightweight.
"""

from pathlib import Path

import pytest

pytest.importorskip("langsmith")
pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from src.retrieval.generate import (
    _capture_usage,
    _estimate_cost,
    _extract_text,
    _trace_inputs,
)
from src.retrieval.prompt import (
    PromptFigure,
    _build_message_content,
    _figure_blocks,
    build_messages,
    prompt_stats,
    select_figures,
)


class _FakeMsg:
    def __init__(self, usage_metadata=None):
        self.usage_metadata = usage_metadata
        self.content = ""


def test_capture_usage_adds_each_chunk():
    """Gemini chunks carry the tokens each chunk added; the totals are their sum."""
    usage = {}
    _capture_usage(_FakeMsg({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}), usage)
    _capture_usage(_FakeMsg({"input_tokens": 0, "output_tokens": 3, "total_tokens": 3}), usage)
    assert usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_capture_usage_single_final_chunk():
    """OpenAI-style streams send the whole count once, on the last chunk."""
    usage = {}
    _capture_usage(_FakeMsg(), usage)
    _capture_usage(_FakeMsg({"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}), usage)
    assert usage == {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}


def test_capture_usage_ignores_non_int():
    usage = {}
    _capture_usage(_FakeMsg({"input_tokens": "nope", "output_tokens": 1, "total_tokens": 10}), usage)
    assert usage == {"output_tokens": 1, "total_tokens": 10}


def test_capture_usage_no_metadata():
    usage = {}
    _capture_usage(_FakeMsg(), usage)
    assert usage == {}


def test_estimate_cost_uses_the_models_own_prices():
    """GPT-4o is priced as GPT-4o (2.50 in, 10.00 out per 1M), not as the default model."""
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    assert _estimate_cost("openai", "gpt-4o", usage) == 3.5
    assert _estimate_cost("gemini", "gemini-2.5-flash-lite", usage) == 0.14


def test_estimate_cost_unknown_model_is_none():
    assert _estimate_cost("openai", "not-a-model", {"input_tokens": 10}) is None


def test_extract_text_string():
    assert _extract_text("plain") == "plain"


def test_extract_text_list_of_parts():
    parts = [{"type": "text", "text": "a"}, {"type": "content", "text": "b"}, "c"]
    assert _extract_text(parts) == "abc"


def test_extract_text_other():
    assert _extract_text(None) == "None"


def test_prompt_stats_counts_text_and_images():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "image_url", "image_url": {"url": "y"}},
    ]
    chars, images = prompt_stats(content)
    assert chars == 5
    assert images == 2


def test_trace_inputs_keep_the_question_and_drop_objects():
    """Only plain values reach LangSmith; the chat model (and its key) never does."""
    recorded = _trace_inputs(
        {
            "retriever": object(),
            "query": "LM555 pins?",
            "llm": object(),
            "model_id": "gemini-3.5-flash-lite",
            "chunks": ["a", "b"],
            "max_images": 0,
            "send_images": True,
        }
    )
    assert recorded == {"question": "LM555 pins?", "max_images": 0, "send_images": True, "num_chunks": 2}


def test_generation_trace_records_figure_labels_not_bytes():
    figures = [PromptFigure("IMG1", 1, 0, "id-1", "lm317", Path("dbv2/images/a.jpg"))]
    recorded = _trace_inputs({"query": "q", "chunks": ["a"], "figures": figures})
    assert recorded == {"question": "q", "num_chunks": 1, "figures": ["IMG1"]}


@pytest.fixture(autouse=True)
def _figures_root(monkeypatch, tmp_path):
    """Figure paths resolve under ``tmp_path``, so no test reads ``dbv2/images``."""
    import src.documents

    (tmp_path / "dbv2" / "images").mkdir(parents=True)
    monkeypatch.setattr(src.documents, "_PROJECT_ROOT", tmp_path)
    return tmp_path


def _chunk(
    summary: str,
    images: list[str],
    source: str | None = None,
    record_id: str | None = None,
    raw_text: str = "",
    tables: list[str] | None = None,
) -> Document:
    """A stored chunk whose figure files hold the given strings as bytes."""
    import src.documents

    paths = []
    for name in images:
        path = f"dbv2/images/{name}.jpg"
        (src.documents._PROJECT_ROOT / path).write_bytes(name.encode())
        paths.append(path)
    return Document(
        id=record_id,
        page_content=summary,
        metadata={"raw_text": raw_text, "tables_html": tables or [], "image_paths": paths}
        | ({"source": source} if source else {}),
    )


def _b64(text: str) -> str:
    import base64

    return base64.b64encode(text.encode()).decode()


def _text(content: list[dict]) -> str:
    return "\n".join(part["text"] for part in content if part["type"] == "text")


def _image_blocks(figures) -> list[dict]:
    return [block for figure in figures for block in _figure_blocks(figure)]


def test_every_retrieved_chunk_reaches_the_prompt():
    """No hidden limit: the prompt holds as many chunks as were retrieved."""
    chunks = [_chunk(f"summary {i}", []) for i in range(1, 8)]
    prompt = _text(_build_message_content(chunks, "q", []))
    assert "[C7] Document: unknown document" in prompt and "summary 7" in prompt
    assert "[C8]" not in prompt


def test_each_chunk_carries_its_document_summary_text_and_tables():
    """The original text goes to the model next to the summary, under the document's name."""
    chunk = _chunk("Summary of pins", [], source="lm317", raw_text="Pin 1 ADJ", tables=["<table>t</table>"])
    prompt = _text(_build_message_content([chunk], "pins?", []))
    assert prompt.index("CONTEXT") < prompt.index("[C1] Document: lm317") < prompt.index("QUESTION: pins?")
    assert "Summary:\nSummary of pins" in prompt
    assert "Text:\nPin 1 ADJ" in prompt
    assert "Table" not in prompt, "a table with no cells is skipped"


def test_tables_reach_the_answer_prompt_as_markdown():
    table = "<table><tr><th>Pin</th><th>Name</th></tr><tr><td>1</td><td>ADJ</td></tr></table>"
    prompt = _text(_build_message_content([_chunk("s", [], tables=[table])], "q", []))
    assert "Table 1:\n| Pin | Name |\n|---|---|\n| 1 | ADJ |" in prompt
    assert "<table>" not in prompt


def test_a_text_only_chunk_is_written_once():
    """Its summary is its original text, so the prompt does not repeat it."""
    chunk = _chunk("Pin 1 ADJ", [], raw_text="Pin 1 ADJ")
    prompt = _text(_build_message_content([chunk], "q", []))
    assert prompt.count("Pin 1 ADJ") == 1 and "Summary:" not in prompt


def test_figures_follow_their_own_chunk():
    chunks = [_chunk("a", ["A1"], source="lm317"), _chunk("b", [], source="ne555"), _chunk("c", ["C1"], source="ne555")]
    content = _build_message_content(chunks, "q", select_figures(chunks, 0))
    order = [part["text"].split("\n")[0] if part["type"] == "text" else "image" for part in content]
    assert order == [
        "CONTEXT",
        "[C1] Document: lm317",
        "[IMG1] Image from lm317, chunk C1",
        "image",
        "[C2] Document: ne555",
        "[C3] Document: ne555",
        "[IMG2] Image from ne555, chunk C3",
        "image",
        "QUESTION: q",
    ]


def test_system_message_holds_the_rules():
    """Grounding and one-document rules always; the figure rule only with figures."""
    chunks = [_chunk("a", ["A1"])]
    with_figures = build_messages(chunks, "q", select_figures(chunks, 0))
    without = build_messages(chunks, "q", [])
    assert [type(m).__name__ for m in with_figures] == ["SystemMessage", "HumanMessage"]
    rules = with_figures[0].content
    assert "using only the context provided" in rules
    assert "do not state it" in rules
    assert "never combine values from different documents" in rules
    assert "cite the chunks it comes from" in rules and "[C1]" in without[0].content
    assert "[IMG1]" in rules and "IMG1" not in without[0].content
    assert "not the document's own figure numbers" in rules


def _images(blocks: list[dict]) -> list[str]:
    return [block["image_url"]["url"] for block in blocks if block["type"] == "image_url"]


def _labels(blocks: list[dict]) -> list[str]:
    return [block["text"] for block in blocks if block["type"] == "text"]


def test_image_cap_takes_images_in_chunk_order():
    """Chunk 1's images come first, then chunk 2's, until the cap."""
    chunks = [_chunk("a", ["A1", "A2"]), _chunk("b", ["B1", "B2"]), _chunk("c", ["C1"])]
    assert _images(_image_blocks(select_figures(chunks, 3))) == [f"data:image/jpeg;base64,{_b64(x)}" for x in ("A1", "A2", "B1")]


def test_image_cap_zero_sends_every_image():
    chunks = [_chunk("a", ["A1", "A2"]), _chunk("b", ["B1"])]
    assert len(_images(_image_blocks(select_figures(chunks, 0)))) == 3


def test_each_figure_is_labelled_just_before_its_image():
    """Labels number figures in prompt order and name the document and chunk."""
    chunks = [_chunk("a", ["A1"], source="lm317"), _chunk("b", ["B1", "B2"], source="ne555")]
    blocks = _image_blocks(select_figures(chunks, 0))
    assert [block["type"] for block in blocks] == ["text", "image_url"] * 3
    assert _labels(blocks) == [
        "[IMG1] Image from lm317, chunk C1",
        "[IMG2] Image from ne555, chunk C2",
        "[IMG3] Image from ne555, chunk C2",
    ]


def test_unreadable_figure_takes_no_label_or_slot(_figures_root):
    """A missing file is skipped: the next figure becomes IMG1 and fills the cap."""
    chunk = _chunk("a", ["A1", "A2"])
    (_figures_root / "dbv2" / "images" / "A1.jpg").unlink()
    blocks = _image_blocks(select_figures([chunk], 1))
    assert _labels(blocks) == ["[IMG1] Image from unknown document, chunk C1"]
    assert _images(blocks) == [f"data:image/jpeg;base64,{_b64('A2')}"]


def test_text_only_model_gets_no_images_and_no_mention_of_them():
    chunks = [_chunk("a", ["A1", "A2"]), _chunk("b", ["B1"])]
    system, user = build_messages(chunks, "q", [])
    assert {part["type"] for part in user.content} == {"text"}
    assert "figure" not in (system.content + _text(user.content)).lower()


def test_selected_figures_know_their_chunk_position_and_record():
    """Each chosen figure can be turned back into its stable image URL."""
    chunks = [_chunk("a", [], record_id="r1"), _chunk("b", ["B1", "B2"], source="ne555", record_id="r2")]
    figures = select_figures(chunks, 0)
    assert [(f.label, f.chunk_number, f.position, f.record_id, f.source) for f in figures] == [
        ("IMG1", 2, 0, "r2", "ne555"),
        ("IMG2", 2, 1, "r2", "ne555"),
    ]


def test_skipped_figure_keeps_the_next_one_at_its_stored_position(_figures_root):
    """A missing first file shifts the label, never the position in the URL."""
    chunk = _chunk("a", ["A1", "A2"], record_id="r1")
    (_figures_root / "dbv2" / "images" / "A1.jpg").unlink()
    (figure,) = select_figures([chunk], 0)
    assert (figure.label, figure.position) == ("IMG1", 1)


def test_trace_attaches_only_the_sent_figures_by_label():
    from types import SimpleNamespace

    from src.retrieval.generate import _attach_figures_to_trace

    chunks = [_chunk("a", ["A1", "A2"])]
    run_tree = SimpleNamespace(attachments=None)
    _attach_figures_to_trace(run_tree, select_figures(chunks, 1))
    assert run_tree.attachments == {"IMG1": ("image/jpeg", b"A1")}


def test_choosing_figures_reads_no_file(monkeypatch):
    """Bytes are read for the model request and the trace, not when choosing."""
    chunks = [_chunk("a", ["A1"])]

    def opened(path):
        raise AssertionError(f"select_figures read {path}")

    monkeypatch.setattr(Path, "read_bytes", opened)
    assert [figure.label for figure in select_figures(chunks, 0)] == ["IMG1"]


def test_figure_outside_the_images_folder_is_not_sent(_figures_root):
    """A stored path is used only if it lies inside IMAGES_DIR, as for the image endpoint."""
    (_figures_root / "secret.txt").write_bytes(b"secret")
    chunk = Document(page_content="a", metadata={"image_paths": ["secret.txt"]})
    assert select_figures([chunk], 0) == []
