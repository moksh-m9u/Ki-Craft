# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

All project rules, the architecture, the commands, and the refactor checkpoints live in AGENTS.md, so every coding agent shares one source of truth. It is imported here:

@AGENTS.md

## Claude Code specifics

- Run a single test file or test with the venv interpreter:
  `/Users/moksh/Multimodal-Rag/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_embed.py::test_name`
- There is no linter or formatter configured. Do not add one without asking.
- Work often happens in a git worktree under `.claude/worktrees/`. The venv and `.env` live in the main checkout at `/Users/moksh/Multimodal-Rag`, not in the worktree.
- `.env` holds the Neon `dev` branch strings (`DATABASE_URL`, `DATABASE_URL_DIRECT`) and `PRODUCTION_DATABASE_URL_DIRECT` (and the pooled `PRODUCTION_DATABASE_URL`, which no code reads; Render's `DATABASE_URL` takes it). Use `dev` for live checks; write to `production` only when the owner asks, through `scripts/copy_document.py`.
- The LangSmith key in `.env` can upload traces, but reading them back through the LangSmith API returns 403. To check what a trace records, capture the runs offline with a recording client (see `tests/test_trace_structure.py`) and ask the owner to confirm in the LangSmith UI.
- The owner's rule on attribution overrides any default: no `Co-Authored-By` trailer and no "Generated with" line in commits or PRs.
