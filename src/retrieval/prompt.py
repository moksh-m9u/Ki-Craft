"""Answer prompt: the messages sent to the chat model for one question.

Sits between retrieval and generation. ``src/retrieval/generate.py`` passes
the retrieved chunks here and gets back a system message with the answering
rules and a user message holding every chunk (its document, summary, original
text and tables as Markdown), each chosen figure right after its chunk, and
the question last. Nothing here calls a model or the network; figure files
are read from disk only when the message is built.

Chunks are labelled ``C1``, ``C2``, ... and images ``IMG1``, ``IMG2``, ..., so
the answer can cite its evidence as ``[C1]`` or ``[IMG1]``. The image labels
deliberately avoid the word "Figure": the chunk text quotes the datasheet's
own captions ("Figure 3. Output Voltage ..."), and a label like "Figure F3"
next to them invites the model to mix the two numberings.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.chunk import Chunk
from src.figures import chunk_figure_file
from src.tables import table_to_markdown


@dataclass(frozen=True)
class PromptFigure:
    """One figure sent to the model, labelled ``IMG<n>`` in prompt order.

    Attributes:
        label: ``IMG1``, ``IMG2``, ... as the prompt names it.
        chunk_number: 1-based position of its chunk in the retrieved list.
        position: 0-based index into that chunk's stored figures.
        record_id: The chunk's store record id, or None if it has none.
        source: The chunk's document name.
        path: The figure file, inside ``IMAGES_DIR``. Its bytes are read only
            for the model request and the trace attachment.
    """

    label: str
    chunk_number: int
    position: int
    record_id: str | None
    source: str
    path: Path


def prompt_stats(message_content: list[dict]) -> tuple[int, int]:
    """Return ``(text_characters, image_count)`` of a message content list."""
    text_characters = sum(len(x["text"]) for x in message_content if x["type"] == "text")
    images = sum(1 for x in message_content if x["type"] == "image_url")
    return text_characters, images


_RULES = (
    "You answer questions about electronics documents using only the context provided. "
    "If the context does not contain the answer, say that the documents provided do not state it. "
    "Each chunk names its document; never combine values from different documents "
    "unless the question asks for a comparison. "
    "Give values with their units and test conditions exactly as stated. "
    "Each chunk starts with its label, like [C1]. After each statement, cite the "
    "chunks it comes from in square brackets, like [C1] or [C1, C3]."
)


_FIGURE_RULE = (
    " Images are labelled IMG1, IMG2, and so on. These labels are not the document's own "
    "figure numbers: a chunk's text may mention \"Figure 3\", but that is not IMG3. "
    "When you use an image, cite its label in square brackets, like [IMG1]. If you name a "
    "document's figure, add the image label only when that image is the figure, "
    "like \"Figure 3 [IMG1]\"."
)


def _system_prompt(with_figures: bool) -> str:
    """The fixed rules; the figure rule only when figures are sent."""
    return _RULES + (_FIGURE_RULE if with_figures else "")


def chunk_label(chunk_number: int) -> str:
    """The prompt label of the chunk at 1-based ``chunk_number``: ``C1``, ``C2``, ..."""
    return f"C{chunk_number}"


def figure_label(figure_number: int) -> str:
    """The prompt label of the 1-based ``figure_number``-th image sent: ``IMG1``, ..."""
    return f"IMG{figure_number}"


def _chunk_text(chunk: Chunk, chunk_number: int) -> str:
    """One chunk as prompt text: its document, summary, original text and tables.

    Tables are sent as Markdown (see ``src/tables.py``).

    A chunk with no tables or figures was stored with its original text as
    its summary, so the text is written once.
    """
    source = chunk.source or "unknown document"
    summary = (chunk.summary or "").strip()
    raw_text = (chunk.raw_text or "").strip()
    lines = [f"[{chunk_label(chunk_number)}] Document: {source}"]
    if summary and summary != raw_text:
        lines.append(f"Summary:\n{summary}")
    if raw_text:
        lines.append(f"Text:\n{raw_text}")
    tables = [markdown for markdown in map(table_to_markdown, chunk.tables_html) if markdown]
    for number, table in enumerate(tables, start=1):
        lines.append(f"Table {number}:\n{table}")
    return "\n\n".join(lines)


def select_figures(chunks: list[Chunk], max_images: int) -> list[PromptFigure]:
    """Choose the figures to send to the model, in chunk order.

    Figures are numbered in prompt order. No figure file is read here. A
    figure whose file is missing, or outside ``IMAGES_DIR``, takes no label
    and does not count toward ``max_images``.

    Args:
        chunks: Retrieved document chunks.
        max_images: Most figures to choose; 0 chooses every readable one.

    Returns:
        The chosen figures, labelled ``IMG1``, ``IMG2``, ...
    """
    figures: list[PromptFigure] = []
    for chunk_number, chunk in enumerate(chunks, start=1):
        source = chunk.source or "unknown document"
        for position in range(len(chunk.image_paths)):
            path = chunk_figure_file(chunk, position)
            if path is None:
                continue
            figures.append(
                PromptFigure(figure_label(len(figures) + 1), chunk_number, position, chunk.record_id, source, path)
            )
            if len(figures) == max_images:
                return figures
    return figures


def _figure_blocks(figure: PromptFigure) -> list[dict]:
    """A figure's label, then its image read from disk as base64."""
    b64 = base64.b64encode(figure.path.read_bytes()).decode()
    return [
        {
            "type": "text",
            "text": f"[{figure.label}] Image from {figure.source}, chunk {chunk_label(figure.chunk_number)}",
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]


def _build_message_content(chunks: list[Chunk], query: str, figures: list[PromptFigure]) -> list[dict]:
    """Build the user message: every chunk with its figures right after it, then the question.

    The context comes first and the question last, which suits long
    multimodal prompts. Every retrieved chunk is included (``top_k`` sets
    the context size); only the figures in ``figures`` are attached.
    """
    content: list[dict] = [{"type": "text", "text": "CONTEXT"}]
    for chunk_number, chunk in enumerate(chunks, start=1):
        content.append({"type": "text", "text": _chunk_text(chunk, chunk_number)})
        for figure in figures:
            if figure.chunk_number == chunk_number:
                content.extend(_figure_blocks(figure))
    content.append({"type": "text", "text": f"QUESTION: {query}"})
    return content


def build_messages(chunks: list[Chunk], query: str, figures: list[PromptFigure]) -> list[BaseMessage]:
    """The system rules and the user message sent to the model."""
    return [
        SystemMessage(content=_system_prompt(bool(figures))),
        HumanMessage(content=_build_message_content(chunks, query, figures)),
    ]
