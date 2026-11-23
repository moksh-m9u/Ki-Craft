"""Check that Postgres search returns what Chroma search returns (Checkpoint 6).

Before the API switches from Chroma to Postgres, this runs the same questions
through both and compares the chunks they pick, by record id. Each question
is embedded once (watsonx) and that one vector goes to both stores, so any
difference comes from the search itself:

* nearest 20: Chroma's ``similarity_search_by_vector`` against
  ``src/retrieval/search.py:nearest_chunks`` (pgvector ``<=>``).
* final 5 after MMR: Chroma's ``max_marginal_relevance_search_by_vector``
  against ``search_postgres(..., "mmr", 5)``, with the same pool size and
  ``lambda_mult`` from ``config/settings.py``.

The questions are the golden set's (``evals/golden_set.json``). It prints one
line per question that differs and a summary, and exits 1 if any differ.
Chroma's index is approximate (HNSW) and pgvector's scan is exact, so a rare
swap at rank 19 or 20 is possible and is reported rather than hidden.

It reads ``DATABASE_URL`` (the pooled string the API will use) and opens
Chroma, which rewrites tracked files; run ``git checkout -- dbv2/chroma_db``
afterwards. The script is deleted with Chroma at the end of Checkpoint 6.

Usage:
    python -m scripts.compare_search
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from config.settings import DATABASE_URL, RETRIEVAL_FETCH_K, RETRIEVAL_LAMBDA_MULT
from src.db.connection import create_db_engine
from src.embed import load_embedding_model
from src.retrieval.search import load_vector_store, nearest_chunks, search_postgres

GOLDEN_SET = Path(__file__).resolve().parent.parent / "evals" / "golden_set.json"
FINAL_K = 5


def compare(chroma_ids: list[str], postgres_ids: list[str]) -> str | None:
    """Describe how two ranked id lists differ; None when they are identical."""
    if chroma_ids == postgres_ids:
        return None
    if set(chroma_ids) == set(postgres_ids):
        return "same chunks, different order"
    only_chroma = [i for i in chroma_ids if i not in postgres_ids]
    only_postgres = [i for i in postgres_ids if i not in chroma_ids]
    return f"only in Chroma {only_chroma}, only in Postgres {only_postgres}"


def main() -> int:
    """Run every golden-set question through both stores and report differences."""
    questions = [q["question"] for q in json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["questions"]]
    embedding = load_embedding_model()
    chroma = load_vector_store(embedding)
    engine = create_db_engine(DATABASE_URL)

    differ = {"nearest 20": 0, "final 5": 0}
    for n, question in enumerate(questions, start=1):
        vector = embedding.embed_query(question)
        results = {
            "nearest 20": (
                [d.id for d in chroma.similarity_search_by_vector(vector, k=RETRIEVAL_FETCH_K)],
                [str(row.id) for row in nearest_chunks(engine, vector, RETRIEVAL_FETCH_K)],
            ),
            "final 5": (
                [
                    d.id
                    for d in chroma.max_marginal_relevance_search_by_vector(
                        vector, k=FINAL_K, fetch_k=RETRIEVAL_FETCH_K, lambda_mult=RETRIEVAL_LAMBDA_MULT
                    )
                ],
                [d.id for d in search_postgres(engine, vector, "mmr", FINAL_K)],
            ),
        }
        for name, (chroma_ids, postgres_ids) in results.items():
            difference = compare(chroma_ids, postgres_ids)
            if difference:
                differ[name] += 1
                print(f"Q{n} {name}: {difference} ({question})")

    print(
        f"{len(questions)} questions: nearest {RETRIEVAL_FETCH_K} differ on {differ['nearest 20']}, "
        f"final {FINAL_K} after MMR differ on {differ['final 5']}"
    )
    return 1 if any(differ.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
