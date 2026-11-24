"""Home page (``/``): what KI-CRAFT does for you, and where to go next.

Written as a product page for someone who reads datasheets, not as a tour of
the code. The copy follows the README; the list of parts is read live from
``GET /v1/documents``. ``app/dashboard.py`` passes in the pages this one links
to, because ``st.switch_page`` needs the registered page object.
"""

from __future__ import annotations

import streamlit as st

from chunk_data import list_document_summaries
from components.diagram import MERMAID_DIAGRAM, mermaid_iframe
from components.example_questions import ask_example, render_example_buttons

GITHUB_URL = "https://github.com/moksh-m9u/Multimodal-Rag"

# (title, body) of each card in "What it does".
_FEATURES: list[tuple[str, str]] = [
    (
        "Reads the figures and tables, not just the text",
        "Each datasheet is split into sections that keep a figure or table with the text "
        "around it. For a question, the model gets the 5 most relevant sections' original "
        "text and tables, and up to 3 of their figures as images.",
    ),
    (
        "Cites every statement",
        "Each statement ends with the sections it came from, as [C1], and the figures, as "
        "[IMG1]. The server checks every label against what the model was actually given "
        "and flags any that match nothing. Cited figures appear right above the answer.",
    ),
    (
        "Says when the datasheet does not",
        "It answers only from the retrieved sections. When they do not state something it "
        "says so instead of filling the gap, keeps units and test conditions with every "
        "value, and never mixes values from two different parts.",
    ),
    (
        "Your choice of model",
        "Gemini 3.5 Flash-Lite by default. Any listed Gemini, OpenAI, Groq or Hugging Face "
        "model works with your own API key, which is sent with the question and not stored.",
    ),
]

_AUDIENCE: list[tuple[str, str]] = [
    ("KiCad users", "tired of cross-checking every pin, capacitor value and footnote against a 60-page PDF."),
    ("Hardware engineers", "checking a design: dropout at a given load, the output voltage formula, the recommended layout."),
    ("Students and makers", "learning to read datasheets: what a pin does, typical versus maximum, what a graph shows."),
    ("Teams", "who want an open alternative to closed AI EDA tools, one they can run, inspect and extend."),
]

# What each indexed part is, for the "Ready to query" list. A document missing
# here is still listed, by name only. Ingestion does not store a description
# yet; once uploads land (Checkpoint 7) this can come from the API.
_PART_KINDS: dict[str, str] = {
    "AN699": "Application note: anti-aliasing filters for data acquisition",
    "LM2596": "3 A step-down switching regulator, 150 kHz",
    "LM317": "Adjustable linear regulator, 1.25 V to 37 V, 1.5 A",
    "LM555": "Timer, astable and monostable",
    "NE555": "Precision timer, astable and monostable",
    "TLV1117": "Low-dropout linear regulator, 800 mA",
}

# (step, what happens) for each stage of "How it works".
_INGESTION_STEPS: list[tuple[str, str]] = [
    ("Extract", "unstructured (hi_res) splits the PDF into text, tables (as HTML) and figures (as images); page headers, footers and logos are dropped."),
    ("Section", "Elements are grouped by heading into sections of at most 3,000 characters, so a figure stays with the text that explains it."),
    ("Summarise", "A vision model reads each section's text, tables and figures and writes a plain-text summary."),
    ("Index", "IBM watsonx Granite turns each summary into a 768-number vector. Postgres with pgvector stores it with the section's original text, tables and figure paths."),
]
_QUERY_STEPS: list[tuple[str, str]] = [
    ("Search", "The question is embedded with the same model. MMR takes the 20 nearest summaries and keeps 5 that are relevant but not repetitive."),
    ("Prompt", "The model gets those 5 sections' original text and tables, labelled C1 to C5, and up to 3 of their figures, labelled IMG1 to IMG3."),
    ("Answer", "Gemini 3.5 Flash-Lite, or your model, writes the answer and cites [C1] and [IMG1]. It streams to the page as it is written."),
    ("Check", "The server reads every citation and keeps only labels the model was given; any other label is shown as a warning."),
]

# (phase, goal, status)
_ROADMAP: list[tuple[str, str, str]] = [
    ("1. Datasheet assistant", "Trustworthy answers from datasheets: text, tables, figures, cited", "In progress"),
    ("2. Schematic review", "Check a KiCad schematic against the datasheets of its parts", "Planned"),
    ("3. Design agent", "Draft schematics in KiCad from requirements, assist with layout", "Planned"),
]


def _cards(items: list[tuple[str, str]]) -> str:
    """Raw HTML for a grid of title-and-text cards, styled by ``styles.css``.

    Fragments are joined with no blank lines: Markdown ends a raw HTML block
    at the first blank line and would print the rest as text.
    """
    body = "".join(f'<div class="kc-card"><h4>{title}</h4><p>{text}</p></div>' for title, text in items)
    return f'<div class="kc-grid">{body}</div>'


def _steps(title: str, steps: list[tuple[str, str]]) -> str:
    """Raw HTML for one numbered stage of the pipeline."""
    rows = "".join(
        f'<li><span class="kc-step-name">{name}</span>{text}</li>' for name, text in steps
    )
    return f'<div class="kc-card"><h4>{title}</h4><ol class="kc-steps">{rows}</ol></div>'


def _render_hero(chat_page: st.Page, explorer_page: st.Page) -> None:
    """The name, the one-line promise, the pitch and the three main actions."""
    st.markdown(
        '<div class="kc-hero">'
        '<div class="kc-eyebrow">Open-source datasheet assistant for KiCad users</div>'
        '<div class="kc-title">KI-CRAFT</div>'
        '<div class="kc-tagline">Ask your datasheets. See the section and figure behind every answer.</div>'
        '<p class="kc-lede">KI-CRAFT reads datasheets and application notes the way an engineer '
        "does, text, tables and figures together, and answers with citations you can check. "
        "It is the first step of an open-source, AI-assisted PCB design toolkit built to work "
        "alongside KiCad.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container(horizontal=True, gap="small"):
        if st.button("Ask a datasheet", type="primary", icon=":material/chat:"):
            st.switch_page(chat_page)
        if st.button("Browse the datasheets", icon=":material/menu_book:"):
            st.switch_page(explorer_page)
        st.link_button("Source on GitHub", GITHUB_URL, icon=":material/code:")


def _render_parts(chat_page: st.Page) -> None:
    """The parts you can ask about, read live from the API, and example questions."""
    st.markdown("## Ready to query")
    try:
        documents = list_document_summaries()
    except Exception as exc:
        st.warning(f"The list of parts is unavailable: {exc}")
        return
    if not documents:
        st.info("No datasheets are indexed yet.")
        return
    parts = []
    for document in documents:
        name = document["id"].upper()
        kind = _PART_KINDS.get(name, "")
        parts.append(f'<div class="kc-part"><b>{name}</b>{f"<span>{kind}</span>" if kind else ""}</div>')
    st.markdown(f'<div class="kc-parts">{"".join(parts)}</div>', unsafe_allow_html=True)

    st.markdown("**Try asking** (click one to ask it):")
    clicked = render_example_buttons("home")
    if clicked:
        ask_example(clicked)
        st.switch_page(chat_page)


def _render_how_it_works() -> None:
    """The two stages of the pipeline, step by step, then the interactive diagram."""
    st.markdown("## How it works")
    st.markdown(
        '<div class="kc-grid">'
        + _steps("Once per datasheet", _INGESTION_STEPS)
        + _steps("For every question", _QUERY_STEPS)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "The summaries are only used to find sections; the model always answers from the "
        "original text, tables and figures. Numbers are the defaults: a request can ask for "
        "more sections or figures."
    )
    # Drawn in the open: inside a collapsed expander the iframe has no width
    # when the diagram is laid out, and it opens zoomed onto an empty corner.
    st.iframe(mermaid_iframe(MERMAID_DIAGRAM), height=560)


def _render_limits() -> None:
    """What it does not do yet, so nobody finds out the hard way."""
    st.markdown("## Current limits")
    st.markdown(
        "- Only the parts listed above can be queried; uploading your own PDF comes next.\n"
        "- A question is searched across every part at once. Name the part in the question "
        "to keep the answer on it.\n"
        "- Citations point to a section and a figure, not yet to a page or a highlighted line."
    )


def _render_roadmap() -> None:
    """Where the project is going, as in the README's status table."""
    st.markdown("## Roadmap")
    st.table(
        [{"Phase": phase, "Goal": goal, "Status": status} for phase, goal, status in _ROADMAP]
    )


def render(chat_page: st.Page, explorer_page: st.Page) -> None:
    """Draw the home page.

    Args:
        chat_page: The Query page, for "Ask a datasheet" and the example questions.
        explorer_page: The Chunk Explorer page, for "Browse the datasheets".
    """
    _render_hero(chat_page, explorer_page)
    st.markdown("## What it does")
    st.markdown(_cards(_FEATURES), unsafe_allow_html=True)
    _render_parts(chat_page)
    _render_how_it_works()
    st.markdown("## Who it is for")
    st.markdown(_cards(_AUDIENCE), unsafe_allow_html=True)
    _render_limits()
    _render_roadmap()
    st.caption(
        f"KI-CRAFT is open source under the Apache 2.0 license. "
        f"[Source]({GITHUB_URL}) · [API reference]({GITHUB_URL}/blob/main/docs/API.md) · "
        f"[Contributing]({GITHUB_URL}/blob/main/CONTRIBUTING.md)"
    )
