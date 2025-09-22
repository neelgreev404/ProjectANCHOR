# Project Anchor: Codebase Guide

A warehouse-native project for evidence-linked multimodal Q and A in BigQuery. One warehouse, one CLI, no glue code.

This README pairs with the writeup in `paper.md`. If you want the story and motivation, read the paper. If you want to run it and understand the files, start here.


## Problem Statement 
Answering simple questions is slow because the proof is scattered across PDFs, screenshots, and images. The goal is a BigQuery‑native flow that ingests public, mixed‑format data and returns a short, cited answer — with validators and a small forecast — using only SQL and built‑in BigQuery AI.

## Impact Statement
Minutes to first cited answer, not hours. Citations and prompt provenance make results auditable. DRY RUN guards and right‑sized vector indexes keep costs in check. The flow is idempotent and extendable to new domains with no external services.


## Who this is for
- You want to try evidence-linked Q and A with citations inside BigQuery.
- You prefer SQL and a clean CLI over a stack of services.
- You need a clear map of the code and a short path to “it runs.”


## Quickstart (about ten minutes)
1) Set environment variables (PowerShell examples):
   - `PROJECT_ID`, `REGION`, `DATASET_ID` (example: `anchor_demo`)
   - Optional: `STAGING_DATASET_ID` and `BQ_CONNECTION_NAME` (example: `your-project.US.gemini`)
   - If it breaks, it is usually your dataset/connection — double check those first.
2) Install packages:
   - `pip install google-cloud-bigquery python-dotenv`
3) Run the CLI (smoke):
   - `python -m src.main --mode test --query "What does the sample image describe?"`
4) Run the full flow (validators, retrieval, forecast):
   - `python -m src.main --mode eval --query "What does the sample image describe?" --kpi orders`
5) Optional — print the benchmark report:
   - `python scripts/bench.py --seed-kpi --query "What does the sample image describe?" --kpi orders`


## Repo layout
```
src/
  mskg/
    core.py          # configuration, constants, BigQuery client, smoke()
    builders.py      # SQL builders (extraction, embeddings, retrieval, validators, answers, forecast, DDL)
    orchestrator.py  # assembles end-to-end pipeline SQL in safe order
    __init__.py      # re-exports for short imports
  sql/
    templates.sql    # consolidated SQL templates (answers, embedding, facts, validators, forecast, index)
    index.sql        # index-only script for creating/rebuilding vector indexes
  main.py            # CLI entry to run the pipeline (test/eval/benchmark modes)
scripts/
  bench.py           # prints a full benchmark report (env, counts, indexes, timings, previews)
  format_code.py     # one-shot formatter (Black + Ruff)
  fill_cred_secret.ps1
```


## Files, explained (with just enough detail)
### `src/mskg/core.py`
- What: `Config`, `AllConstants`, `get_bq_client()`, and a tiny `smoke()`.
- Why: keeps knobs in one place; easy client creation.
- Use when: you need a client, want to tweak models/dims, or sanity-check the env.


### `src/mskg/builders.py` (the one you will live in)
- What: builds the SQL strings for the whole flow — descriptors, embeddings, retrieval, validators, answer synthesis, forecast, and canonical DDL.
- Why: keeps all SQL together and idempotent. One clean `MERGE` per write, stable keys, no temp tables.
- Use when: you want to change prompts, switch models, adjust k/context, or add a validator.
- Notes:
  - Retrieval: deterministic VECTOR_SEARCH plumbing that stays fast and cheap.
  - Validators: lightweight on purpose; cheap signals like ColorMatch and ValueCheck.


### `src/mskg/orchestrator.py`
- What: orders statements safely and returns a list via `assemble_pipeline_sql(cfg, cst)`.
- Why: fewer surprises; right statements in the right order.
- Use when: you want to run the whole thing as a pipeline or preview the stages.


### `src/main.py`
- What: CLI driver (`--mode test|eval|benchmark`) that executes the pipeline and prints a compact run report.
- Why: terminal-first demo; easy for judges.
- Use when: you want one command to apply DDL, run extract/embeddings/validators, generate an answer, and produce a forecast.


### `scripts/bench.py`
- What: benchmark CLI that prints a full report.
- Why: reproducible end-to-end performance snapshot of this repo in your project.
- Report includes:
  - Environment (project, region, dataset, connection, models, dims/k, horizon)
  - Dataset snapshot (row counts, index health) with query timing/bytes
  - Retrieval, answer, and forecast timings with bytes, cache hit, and slot-ms
  - Latest answer preview (summary, citations, confidence)
  - Forecast summary (rows, date range, p25/p50/p75)
- Run:
  - `python scripts/bench.py --seed-kpi --query "What does the sample image describe?" --kpi orders`


### `scripts/format_code.py`
- What: one-shot Black + Ruff autofix formatter for the repo.
- Why: keep code style consistent; safe and idempotent to rerun.
- Run:
  - `python scripts/format_code.py --path . --line-length 88`


### `src/sql/templates.sql`
- What: a single bundle of templates; easy to read and run pieces by hand.
- Tip: keep variables in sync with `Config`.


### `src/sql/index.sql`
- What: index-only script that creates vector indexes (`text_emb_idx`, `image_emb_idx`) if vectors exist and rebuilds them on growth.
- When: use for a targeted (re)build if you skipped pipeline steps or tuned index settings.
- How: run it in BigQuery (Console or CLI) with `${DATASET_ID}` replaced, or rely on `main.py` which runs index creation automatically.


## How the pieces connect
1) `schemas_ddl_sql` creates tables if missing (and remote models for embeddings).
2) Object URIs → text chunks and image captions.
3) Embeddings for text and image descriptors.
4) Vector indexes created or refreshed when tables grow (automatically in the pipeline, or manually via `src/sql/index.sql`).
5) Facts via `AI.GENERATE` (with `output_schema`) and a single `MERGE`.
6) Retrieval pulls neighbors; validators run on a small sample.
7) `synthesize_answer_sql` writes a short answer with citations.
8) `forecast_sql` produces a tiny forward curve.


## Design choices (and small performance notes)
- Minimalism: one warehouse, one CLI, no glue code.
- Safe re-runs: `MERGE` with stable keys, no blind inserts.
- Cost awareness: dry-run checks; build indexes only when they make sense.
- Evidence first: citations limited to allowed sources; validators add small, cheap signals.
- In my tests: ~10k vectors → retrieval in ~800 ms, answer in ~3 s (US region, caching off, demo scale).


## Common hiccups (and fixes)
- main.py is a little unpolished as it was done at last minute and doesnt adaequately demonstrate the full capabilties of this project.
- Dataset or region mismatch: set `PROJECT_ID`, `REGION`, `DATASET_ID` consistently.
- Empty vectors: filtered before index creation.
- Validators over-fire on edge cases (e.g., color names like “charcoal”). Relax the rule or mark “inconclusive.”
- Index timing: for tiny corpora, skip the index; create it once you have a few thousand vectors.
- Use `scripts/bench.py` for clean timing output.


## FAQ
- Can I add audio or more modalities? Yes, if you can extract descriptors and embed them.
- Does it work without indexes? Yes; it just feels snappier once indexes exist.
- Where do I change models and dims? In `AllConstants` in `core.py`.
- How do I replay prompts? Prompt id, version, model id, and a simple input hash are logged.


## Try it yourself
- Use the CLI: `python -m src.main --mode test --query "What does the sample image describe?"`
- For a full run: `python -m src.main --mode eval --query "…" --kpi orders`
- To print timings: `python scripts/bench.py --seed-kpi --query "…" --kpi orders`


## Contributing
- Keep SQL idempotent and easy to read.
- Prefer one statement per write with a clear key.
- Update `paper.md` when you change the flow.


## License
Public demo code for hackathon use. Replace or extend for your context.

This is hackathon code — expect rough edges. If you extend it, let me know; I would love to see it.
