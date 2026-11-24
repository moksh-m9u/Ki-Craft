"""The project's own chunk type: one stored section of a document.

Every step of the query path passes chunks in this shape: the database reads
build them (``src/db/chunks.py``), retrieval returns them
(``src/retrieval/search.py``), the prompt and the citation check read them
(``src/retrieval/prompt.py``), and the API turns them into response bodies
(``api/responses.py``). Each field is named and typed, so a reader sees what a
chunk holds without knowing which keys a metadata dict happens to carry.

LangChain's ``Document`` appears only at the edges that need it: ingestion
builds Documents before storing, and the LangSmith trace records chunks in
the Document layout so its UI can show them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """One stored chunk with its original content.

    Attributes:
        record_id: The chunk's primary key in Postgres (a UUID string). Figure
            URLs contain it. None only for chunks built in tests.
        source: The document's name, its PDF's file name without ``.pdf``
            (``lm317``).
        chunk_id: Position in the document, counted from 1 at ingestion.
        summary: The vision model's summary; this is the text that was embedded.
        raw_text: The chunk's original text as extracted from the PDF.
        tables_html: The chunk's tables, as HTML.
        image_paths: The chunk's figure files, relative to the repository
            root, in position order. Nothing here reads them.
    """

    record_id: str | None
    source: str
    chunk_id: int
    summary: str
    raw_text: str = ""
    tables_html: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)

    @property
    def has_table(self) -> bool:
        """True when the chunk holds at least one table."""
        return bool(self.tables_html)

    @property
    def has_image(self) -> bool:
        """True when the chunk has at least one figure."""
        return bool(self.image_paths)
