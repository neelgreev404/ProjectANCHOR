#!/usr/bin/env python3
"""
Project Anchor CLI (terminal-first demonstration)

Blueprint driver that orchestrates the end-to-end flow without notebooks.
- Modes: test (quickstart), eval (full run + metrics), benchmark (timings/cost)
- Dry-run cost guards before heavy statements
- Terminal summaries: index health, retrieval metrics, validator coverage,
  row counts, recent answers

Usage examples:
  # quick smoke to a cited answer
  python -m src.main --mode test --query "What does the sample image describe?"

  # full evaluation run with metrics and forecast
  python -m src.main --mode eval --query "What changed since v2?" --kpi demo_kpi

  # benchmark timings and bytes
  python -m src.main --mode benchmark --query "Describe the brochure" --kpi demo_kpi

Environment knobs can be provided directly in the shell or via config/mskg.yaml.
Key envs: PROJECT_ID, REGION, DATASET_ID, STAGING_DATASET_ID, BQ_CONNECTION_NAME.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from google.cloud import bigquery  # type: ignore

from .mskg.core import Config, AllConstants
from .mskg.orchestrator import assemble_pipeline_sql
from .mskg.builders import forecast_sql


_TTY = sys.stdout.isatty()
_SPARK_BARS = ""


class Spinner:
    def __init__(
        self, text: str = "working", *, enabled: bool = True, interval: float = 0.1
    ) -> None:
        self.text = text
        self.enabled = enabled
        self.interval = interval
        self._stop = threading.Event()
        self._frames = (
            ["-", "\\", "|", "/"]
            if os.name == "nt"
            else ["⠋", "⠙", "⠚", "⠞", "⠖", "⠦", "⠴", "⠲", "⠳", "⠓"]
        )
        self._th: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled:
            return

        def _run() -> None:
            i = 0
            while not self._stop.is_set():
                frame = self._frames[i % len(self._frames)]
                print(f"{frame} {self.text}    ", end="\r", flush=True)
                i += 1
                time.sleep(self.interval)

        self._th = threading.Thread(target=_run, daemon=True)
        self._th.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=0.2)
        print(" " * 80, end="\r", flush=True)

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _spark(values: Iterable[float]) -> str:
    return ""


def _hr(title: str) -> None:
    print(f"\n=== {title} ===")


def _fmt_bytes(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    if mb < 1024:
        return f"{mb:.2f} MB"
    gb = mb / 1024
    return f"{gb:.2f} GB"


def _progress_bar(done: int, total: int, *, width: int = 24) -> str:
    done = max(0, min(done, total))
    fill = int(width * (done / max(1, total)))
    return "[" + ("█" * fill) + ("-" * (width - fill)) + "]"


def _label_from_sql(sql: str) -> str:
    u = sql.upper()

    def has(fragment: str) -> bool:
        return fragment in u

    if has("CREATE TABLE IF NOT EXISTS"):
        return "Apply DDL"
    if has("VECTOR INDEX"):
        return "Ensure vector indexes"
    if has("MERGE") and has(".TEXT_EMB"):
        return "Text embeddings"
    if has("MERGE") and has(".IMAGE_EMB"):
        return "Image embeddings"
    if has("MERGE") and has(".IMAGE_DESC"):
        return "Image captions"
    if has("OBJECTS_PDFS") and has("OUTPUT_SCHEMA => 'TEXT STRING'"):
        return "Extract PDF text"
    if has("OBJECTS_SCREENS") and has("OUTPUT_SCHEMA => 'TEXT STRING'"):
        return "Extract screenshot text"
    if has("OBJECTS_IMAGES") and has("OUTPUT_SCHEMA => 'TEXT STRING'") and has("#IMG"):
        return "Extract image text"
    if has("FACTS_GENERIC") and has("AI.GENERATE_TABLE"):
        return "Facts extraction"
    if has("METRICS_VALIDATORS") and has("GENERATE_BOOL"):
        return "Validator (BOOL)"
    if has("METRICS_VALIDATORS") and has("GENERATE_DOUBLE"):
        return "Validator (DOUBLE)"
    if has("PRECOMPUTED_NEIGHBORS"):
        return "Precompute neighbors"
    if has("EDGES_SEMANTIC"):
        return "Semantic edges"
    if has("METRICS_RETRIEVAL"):
        return "Retrieval metrics"
    if has("INSERT INTO") and has(".ANSWERS"):
        return "Answer synthesis"
    if has("MERGE") and has("KPI_SERIES"):
        return "Aggregate KPI series"
    if has("KPI_FORECAST") and has("AI.FORECAST"):
        return "Forecast"
    if has("CREATE OR REPLACE VIEW") and has("V_KPI_CARDS"):
        return "Create KPI view"
    return "SQL step"


@dataclass
class ExecResult:
    elapsed_ms: float
    total_bytes: int
    cache_hit: bool
    ok: bool
    error: Optional[str] = None


def _make_params(query_text: str, kpi_id: str) -> List[bigquery.ScalarQueryParameter]:
    return [
        bigquery.ScalarQueryParameter("query_text", "STRING", query_text),
        bigquery.ScalarQueryParameter("kpi", "STRING", kpi_id),
    ]


def _dry_run(
    client: bigquery.Client, sql: str, params: List[bigquery.ScalarQueryParameter]
) -> int:
    job_cfg = bigquery.QueryJobConfig()
    job_cfg.dry_run = True
    job_cfg.use_query_cache = False
    if params:
        job_cfg.query_parameters = params
    job = client.query(sql, job_config=job_cfg)
    return int(job.total_bytes_processed or 0)


def _execute(
    client: bigquery.Client,
    sql: str,
    params: List[bigquery.ScalarQueryParameter],
    *,
    do_dry_run: bool,
    cost_cap_bytes: Optional[int],
) -> ExecResult:
    try:
        if do_dry_run:
            est = _dry_run(client, sql, params)
            if cost_cap_bytes is not None and est > cost_cap_bytes:
                return ExecResult(
                    0.0,
                    est,
                    False,
                    False,
                    error=f"DRY RUN cap exceeded: {_fmt_bytes(est)} > {_fmt_bytes(cost_cap_bytes)}",
                )

        t0 = time.perf_counter()
        job_cfg = bigquery.QueryJobConfig()
        if params:
            job_cfg.query_parameters = params
        job = client.query(sql, job_config=job_cfg)
        job.result()
        t1 = time.perf_counter()
        return ExecResult(
            elapsed_ms=(t1 - t0) * 1000.0,
            total_bytes=int(job.total_bytes_processed or 0),
            cache_hit=bool(job.cache_hit),
            ok=True,
        )
    except Exception as e:
        return ExecResult(0.0, 0, False, False, error=str(e))


def _seed_objects(
    client: bigquery.Client, cfg: Config, image_uris: List[str], pdf_uris: List[str]
) -> None:
    if not image_uris and not pdf_uris:
        return

    def _merge_values(table: str, rows: List[Tuple[str, str]]):
        if not rows:
            return
        struct_elems = ",".join(
            [
                f"STRUCT('{u}' AS object_uri, '{mt}' AS media_type, NULL AS object_ref)"
                for (u, mt) in rows
            ]
        )
        sql = (
            f"MERGE {cfg.table(table)} AS dst\n"
            f"USING (SELECT * FROM UNNEST([{struct_elems}])) AS src\n"
            f"ON dst.object_uri = src.object_uri\n"
            f"WHEN NOT MATCHED THEN INSERT (object_uri, media_type, object_ref) VALUES (src.object_uri, src.media_type, src.object_ref)"
        )
        _ = client.query(sql).result()

    img_rows = [(u, "image/jpeg") for u in image_uris]
    pdf_rows = [(u, "application/pdf") for u in pdf_uris]
    _merge_values("objects_images", img_rows)
    _merge_values("objects_pdfs", pdf_rows)


def _print_index_health(client: bigquery.Client, cfg: Config) -> None:
    _hr("Index health")
    sql = f"SELECT * FROM {cfg.table('v_index_health')}"
    rows = list(client.query(sql).result())
    if not rows:
        print("(no data)")
        return
    for r in rows:
        print(
            f"- {r['table_name']}: rows={r['row_count']}, has_index={bool(r['has_index'])}"
        )


def _print_validator_coverage(client: bigquery.Client, cfg: Config) -> None:
    _hr("Validator outcomes")
    sql = f"SELECT rule_id, n, ROUND(pass_rate, 3) AS pass_rate FROM {cfg.table('v_validator_coverage')} ORDER BY rule_id"
    rows = list(client.query(sql).result())
    if not rows:
        print("(no validator rows)")
        return
    for r in rows:
        print(f"- {r['rule_id']}: n={r['n']}, pass_rate={r['pass_rate']}")


def _print_retrieval_metrics(client: bigquery.Client, cfg: Config) -> None:
    _hr("Retrieval metrics")
    sql = f"SELECT k, ROUND(recall_k,3) AS recall_k, ROUND(precision_k,3) AS precision_k, ROUND(mrr,3) AS mrr FROM {cfg.table('v_retrieval_metrics')} ORDER BY k"
    rows = list(client.query(sql).result())
    if not rows:
        print("(no retrieval metrics)")
        return
    for r in rows:
        print(
            f"- k={r['k']}: recall={r['recall_k']}, precision={r['precision_k']}, mrr={r['mrr']}"
        )


def _print_row_counts(client: bigquery.Client, cfg: Config) -> None:
    _hr("Table row counts")
    tables = [
        "image_desc",
        "text_chunks",
        "text_emb",
        "answers",
        "kpi_forecast",
    ]
    for t in tables:
        sql = f"SELECT COUNT(*) AS n FROM {cfg.table(t)}"
        n = list(client.query(sql).result())[0]["n"]
        print(f"- {t}: {n}")


def _print_answer_sample(client: bigquery.Client, cfg: Config) -> None:
    _hr("Latest answer")
    sql = (
        f"SELECT query, summary, ARRAY_TO_STRING(citations, ', ') AS cites, ROUND(confidence,3) AS conf "
        f"FROM {cfg.table('answers')} ORDER BY created_ts DESC LIMIT 1"
    )
    rows = list(client.query(sql).result())
    if not rows:
        print("(no answers yet)")
        return
    r = rows[0]
    print(f"Q: {r['query']}")
    print(f"A: {r['summary']}")
    print(f"Citations: {r['cites']}")
    print(f"Confidence: {r['conf']}")


def _print_forecast_preview(client: bigquery.Client, cfg: Config, kpi: str) -> None:
    _hr("Forecast preview")
    sql = (
        f"SELECT ts, predicted FROM {cfg.table('kpi_forecast')} "
        f"WHERE backtest = FALSE AND kpi_id = @kpi ORDER BY ts LIMIT 48"
    )
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("kpi", "STRING", kpi)]
        ),
    )
    rows = list(job.result())
    if not rows:
        print("(no forecast rows)")
        return
    vals = [float(r["predicted"]) for r in rows]
    print(_spark(vals))


@dataclass
class RunOptions:
    mode: str
    query_text: str
    kpi_id: str
    dry_run_first: bool
    prompts_dir: Optional[str] = None
    images_prefix: Optional[str] = None
    pdfs_prefix: Optional[str] = None
    screens_prefix: Optional[str] = None


def _apply_mode_env(mode: str) -> None:
    if mode == "test":
        os.environ["QUICKSTART"] = "true"
        os.environ["FULL_RUN"] = "false"
        os.environ["EMBED_MAX_ROWS_TEXT"] = "50"
        os.environ["EMBED_MAX_ROWS_IMAGE"] = "25"
        os.environ["FACTS_MAX_ROWS"] = "50"
        os.environ["FACTS_MAX_CHARS"] = "800"
        os.environ["HOLDOUT_SIZE"] = "10"
    elif mode in ("eval", "benchmark"):
        os.environ["QUICKSTART"] = "false"
        os.environ["FULL_RUN"] = "true"
        os.environ["EMBED_MAX_ROWS_TEXT"] = "1500"
        os.environ["EMBED_MAX_ROWS_IMAGE"] = "400"
        os.environ["FACTS_MAX_ROWS"] = "1000"
        os.environ["FACTS_MAX_CHARS"] = "1500"
        os.environ["HOLDOUT_SIZE"] = "100"
    else:
        os.environ["QUICKSTART"] = "true"
        os.environ["FULL_RUN"] = "false"


def run(opts: RunOptions) -> int:
    _apply_mode_env(opts.mode)

    if opts.prompts_dir:
        os.environ["PROMPTS_DIR"] = opts.prompts_dir
    if opts.images_prefix:
        os.environ["GCS_IMAGES_URI_PREFIX"] = opts.images_prefix
        os.environ["OBJECT_TABLES_ENABLED"] = os.environ.get(
            "OBJECT_TABLES_ENABLED", "true"
        )
    if opts.pdfs_prefix:
        os.environ["GCS_PDFS_URI_PREFIX"] = opts.pdfs_prefix
        os.environ["OBJECT_TABLES_ENABLED"] = os.environ.get(
            "OBJECT_TABLES_ENABLED", "true"
        )
    if opts.screens_prefix:
        os.environ["GCS_SCREENS_URI_PREFIX"] = opts.screens_prefix
        os.environ["OBJECT_TABLES_ENABLED"] = os.environ.get(
            "OBJECT_TABLES_ENABLED", "true"
        )

    os.environ.setdefault("TEXT_EMBED_MODEL", "bq/embedding-text-1")
    os.environ.setdefault("IMAGE_EMBED_MODEL", "bq/embedding-image-1")

    cfg = Config.from_env()
    cst = AllConstants()
    cst.validate()
    client = bigquery.Client(project=cfg.project_id, location=cfg.region)

    ddl_sql = assemble_pipeline_sql(cfg, cst)[0]
    _ = client.query(ddl_sql).result()

    image_uris = [
        "gs://cloud-samples-data/bigquery/tutorials/cymbal-pets/images/cat.jpg",
        "gs://cloud-samples-data/bigquery/tutorials/cymbal-pets/images/dog.jpg",
    ]
    pdf_uris = [
        "gs://cloud-samples-data/bigquery/tutorials/cymbal-pets/documents/cymbal_pets_brochure.pdf",
    ]
    try:
        _seed_objects(client, cfg, image_uris=image_uris, pdf_uris=pdf_uris)
    except Exception:
        pass

    statements = assemble_pipeline_sql(cfg, cst)

    if cfg.quickstart:
        try:
            from .mskg.orchestrator import _ensure_one_chunk_from_captions_sql  # type: ignore

            tiny_steps = [
                _ensure_one_chunk_from_captions_sql(cfg),
            ]
            for sql in tiny_steps:
                _ = client.query(sql).result()
        except Exception:
            pass

    try:
        kpi_n = list(
            client.query(
                f"SELECT COUNT(*) AS n FROM {cfg.table('kpi_forecast')} WHERE backtest = FALSE"
            ).result()
        )[0]["n"]
        facts_n = list(
            client.query(
                f"SELECT COUNT(*) AS n FROM {cfg.table('facts_generic')}"
            ).result()
        )[0]["n"]
        if kpi_n == 0 and facts_n > 0 and not cfg.quickstart:
            from .mskg.orchestrator import _kpi_series_from_facts_sql  # type: ignore

            statements.append(_kpi_series_from_facts_sql(cfg))
            statements.append(forecast_sql(cfg, cst, "@kpi"))
    except Exception:
        pass

    params = _make_params(opts.query_text, opts.kpi_id)
    cost_cap_bytes = int(cst.budgets.max_gb_scanned_per_demo) * 1024 * 1024 * 1024

    _hr(f"Running pipeline in mode={opts.mode}")
    totals_elapsed = 0.0
    totals_bytes = 0
    failures = 0
    total = len(statements)
    for i, sql in enumerate(statements, start=1):
        short = _label_from_sql(sql)
        bar = _progress_bar(i - 1, total)
        print(f"{bar} {i-1}/{total} {short}", end="\r", flush=True)

        do_dry = opts.dry_run_first or opts.mode == "benchmark"
        with Spinner(f"{short}", enabled=_TTY):
            res = _execute(
                client, sql, params, do_dry_run=do_dry, cost_cap_bytes=cost_cap_bytes
            )
        if not res.ok:
            print(" " * 80, end="\r")
            print(f"[{i}/{total}] {short}: ERROR: {res.error}")
            failures += 1
            continue
        print(" " * 80, end="\r")
        print(
            f"[{i}/{total}] {short}: {res.elapsed_ms:.0f} ms, bytes={_fmt_bytes(res.total_bytes)}, cache_hit={res.cache_hit}"
        )
        totals_elapsed += res.elapsed_ms
        totals_bytes += res.total_bytes

    _hr("Run totals")
    print(
        f"Elapsed ≈ {totals_elapsed/1000.0:.1f} s, Bytes ≈ {_fmt_bytes(totals_bytes)}, Failures={failures}"
    )

    _print_row_counts(client, cfg)
    _print_index_health(client, cfg)
    _print_retrieval_metrics(client, cfg)
    _print_validator_coverage(client, cfg)
    _print_answer_sample(client, cfg)

    print(
        f"{_progress_bar(len(statements), len(statements))} {len(statements)}/{len(statements)} done"
    )

    return 0 if failures == 0 else 2


def _parse_args(argv: List[str]) -> RunOptions:
    p = argparse.ArgumentParser(description="MSKG CLI: terminal-first demo driver")
    p.add_argument("--mode", choices=["test", "eval", "benchmark"], default="test")
    p.add_argument(
        "--query", dest="query_text", default="What does the sample image describe?"
    )
    p.add_argument("--kpi", dest="kpi_id", default="demo_kpi")
    p.add_argument(
        "--no-dry-run",
        dest="dry_run_first",
        action="store_false",
        help="Skip DRY RUN guard per statement",
    )
    p.set_defaults(dry_run_first=True)
    p.add_argument("--prompts-dir", dest="prompts_dir", default=None)
    p.add_argument("--images-prefix", dest="images_prefix", default=None)
    p.add_argument("--pdfs-prefix", dest="pdfs_prefix", default=None)
    p.add_argument("--screens-prefix", dest="screens_prefix", default=None)
    args = p.parse_args(argv)
    return RunOptions(
        mode=args.mode,
        query_text=args.query_text,
        kpi_id=args.kpi_id,
        dry_run_first=args.dry_run_first,
        prompts_dir=args.prompts_dir,
        images_prefix=args.images_prefix,
        pdfs_prefix=args.pdfs_prefix,
        screens_prefix=args.screens_prefix,
    )


def main(argv: Optional[List[str]] = None) -> int:
    opts = _parse_args(argv or sys.argv[1:])
    try:
        return run(opts)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
