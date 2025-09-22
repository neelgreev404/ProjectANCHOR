#!/usr/bin/env python3
import argparse
import time
import threading
from typing import Tuple, Optional
import sys
from pathlib import Path

from google.cloud import bigquery  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mskg.core import Config, AllConstants, get_bq_client  # type: ignore
from src.mskg.builders import (  # type: ignore
    retrieval_context_sql,
    synthesize_answer_sql,
    forecast_sql,
)


class ProgressSpinner:
    """A simple progress spinner to show activity."""

    def __init__(self, message: str):
        self.message = message
        self.spinning = False
        self.thread = None
        self.chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def start(self):
        """Start the spinner."""
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the spinner and clear the line."""
        self.spinning = False
        if self.thread:
            self.thread.join()
        print(f"\r{' ' * (len(self.message) + 10)}\r", end="", flush=True)

    def _spin(self):
        """Internal spinning animation."""
        i = 0
        while self.spinning:
            print(
                f"\r{self.chars[i % len(self.chars)]} {self.message}...",
                end="",
                flush=True,
            )
            time.sleep(0.1)
            i += 1


def print_progress(step: str, current: int, total: int):
    """Print a progress indicator."""
    percentage = (current / total) * 100
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    print(f"[{current}/{total}] {bar} {percentage:5.1f}% - {step}")


def measure_query_performance(
    client: bigquery.Client,
    sql_text: str,
    *,
    params: Optional[list[bigquery.ScalarQueryParameter]] = None,
    description: str = "Executing query",
) -> Tuple[float, float, bool, float, str]:
    """Run a query and measure execution time and bytes processed."""
    spinner = ProgressSpinner(description)
    spinner.start()

    try:
        start_time = time.perf_counter()

        job_config = bigquery.QueryJobConfig()
        if params:
            job_config.query_parameters = params

        job = client.query(sql_text, job_config=job_config)
        _ = job.result()

        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time

        bytes_processed_mb = 0.0
        cache_hit = False
        total_slot_ms = 0.0
        stmt_type = ""
        try:
            query_stats = job._properties.get("statistics", {}).get("query", {})  # type: ignore[attr-defined]
            if "totalBytesProcessed" in query_stats:
                bytes_processed_mb = float(query_stats["totalBytesProcessed"]) / (
                    1024.0 * 1024.0
                )
            if "cacheHit" in query_stats:
                cache_hit = bool(query_stats["cacheHit"])
            if "totalSlotMs" in query_stats:
                total_slot_ms = float(
                    query_stats["totalSlotMs"]
                )  # slot-ms across workers
            if "statementType" in query_stats:
                stmt_type = str(query_stats["statementType"]) or ""
        except Exception:
            pass

        return elapsed_seconds, bytes_processed_mb, cache_hit, total_slot_ms, stmt_type

    finally:
        spinner.stop()


def build_context_query_with_fallback(
    cfg: Config, cst: AllConstants, query_param: str
) -> str:
    """Build a context retrieval query that falls back to image descriptions if needed."""
    topk = cst.chunk.topk_context
    primary_query = retrieval_context_sql(cfg, cst, query_param, topk)

    return (
        f"WITH ctx_primary AS (\n"
        f"  {primary_query}\n"
        f"), ctx_backup AS (\n"
        f"  SELECT source_uri, caption AS text\n"
        f"  FROM {cfg.table('image_desc')}\n"
        f"  WHERE caption IS NOT NULL\n"
        f"  ORDER BY created_ts DESC\n"
        f"  LIMIT {topk}\n"
        f"), ctx_union AS (\n"
        f"  SELECT * FROM ctx_primary\n"
        f"  UNION ALL\n"
        f"  SELECT * FROM ctx_backup\n"
        f")\n"
        f"SELECT source_uri, text FROM ctx_union LIMIT {topk}"
    )


def get_kpi_seeding_query(cfg: Config) -> str:
    """Generate SQL to seed KPI data from the public thelook dataset."""
    return (
        f"MERGE {cfg.table('kpi_series')} AS target\n"
        f"USING (\n"
        f"  SELECT 'orders' AS kpi_id, DATE(created_at) AS ts, COUNT(*) AS value, ARRAY<STRING>[] AS tags\n"
        f"  FROM `bigquery-public-data.thelook_ecommerce.orders`\n"
        f"  WHERE created_at IS NOT NULL AND DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY)\n"
        f"  GROUP BY ts\n"
        f") AS source\n"
        f"ON target.kpi_id = source.kpi_id AND target.ts = source.ts\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (kpi_id, ts, value, tags) VALUES (source.kpi_id, source.ts, source.value, source.tags);"
    )


def build_counts_query(cfg: Config) -> str:
    t = cfg.table
    return (
        f"SELECT\n"
        f"  (SELECT COUNT(*) FROM {t('objects_images')}) AS objects_images,\n"
        f"  (SELECT COUNT(*) FROM {t('objects_pdfs')}) AS objects_pdfs,\n"
        f"  (SELECT COUNT(*) FROM {t('objects_screens')}) AS objects_screens,\n"
        f"  (SELECT COUNT(*) FROM {t('image_desc')}) AS image_desc,\n"
        f"  (SELECT COUNT(*) FROM {t('text_chunks')}) AS text_chunks,\n"
        f"  (SELECT COUNT(*) FROM {t('text_emb')}) AS text_emb,\n"
        f"  (SELECT COUNT(*) FROM {t('image_emb')}) AS image_emb,\n"
        f"  (SELECT COUNT(*) FROM {t('facts_generic')}) AS facts_generic,\n"
        f"  (SELECT COUNT(*) FROM {t('answers')}) AS answers,\n"
        f"  (SELECT COUNT(*) FROM {t('kpi_series')}) AS kpi_series,\n"
        f"  (SELECT COUNT(*) FROM {t('kpi_forecast')}) AS kpi_forecast,\n"
        f"  (SELECT COUNT(*) FROM {t('metrics_validators')}) AS metrics_validators,\n"
        f"  (SELECT COUNT(*) FROM {t('metrics_retrieval')}) AS metrics_retrieval,\n"
        f"  (SELECT COUNT(*) > 0 FROM {t('INFORMATION_SCHEMA.VECTOR_INDEXES')} WHERE index_name = 'text_emb_idx') AS has_text_idx,\n"
        f"  (SELECT COUNT(*) > 0 FROM {t('INFORMATION_SCHEMA.VECTOR_INDEXES')} WHERE index_name = 'image_emb_idx') AS has_image_idx\n"
    )


def build_answer_preview_query(cfg: Config) -> str:
    return (
        f"SELECT summary, ARRAY_LENGTH(citations) AS n_citations, confidence, created_ts\n"
        f"FROM {cfg.table('answers')}\n"
        f"ORDER BY created_ts DESC\n"
        f"LIMIT 1"
    )


def build_forecast_summary_query(cfg: Config, kpi_id_param: str) -> str:
    return (
        f"SELECT\n"
        f"  COUNTIF(backtest = FALSE) AS rows_prod,\n"
        f"  CAST(MIN(IF(backtest = FALSE, ts, NULL)) AS STRING) AS min_ts,\n"
        f"  CAST(MAX(IF(backtest = FALSE, ts, NULL)) AS STRING) AS max_ts,\n"
        f"  APPROX_QUANTILES(IF(backtest = FALSE, predicted, NULL), 5)[OFFSET(1)] AS p25,\n"
        f"  APPROX_QUANTILES(IF(backtest = FALSE, predicted, NULL), 5)[OFFSET(2)] AS p50,\n"
        f"  APPROX_QUANTILES(IF(backtest = FALSE, predicted, NULL), 5)[OFFSET(3)] AS p75\n"
        f"FROM {cfg.table('kpi_forecast')}\n"
        f"WHERE kpi_id = {kpi_id_param}"
    )


def run() -> int:
    parser = argparse.ArgumentParser(prog="bench", add_help=True)
    parser.add_argument(
        "--query", type=str, default="What does the sample image describe?"
    )
    parser.add_argument("--kpi", type=str, default="orders")
    parser.add_argument(
        "--seed-kpi",
        action="store_true",
        help="Seed KPI data before running benchmarks",
    )
    args = parser.parse_args()

    print("Starting BigQuery Performance Benchmarks")
    print("=" * 50)

    print("Initializing configuration...")
    cfg = Config.from_env()
    cst = AllConstants()

    print("Connecting to BigQuery...")
    client = get_bq_client(cfg)
    print("Connected successfully!")

    # Environment summary
    print("\n--- Environment ---")
    print(f"Project         : {cfg.project_id}")
    print(f"Region          : {cfg.region}")
    print(f"Dataset         : {cfg.dataset_id}")
    print(f"Connection      : {cfg.connection_id_short()}")
    print(f"Text model      : {cst.embed.text_model}")
    print(f"Image model     : {cst.embed.image_model}")
    print(
        f"Vector dims     : text={cst.embed.text_vector_dim}, image={cst.embed.image_vector_dim}"
    )
    print(f"Vector k        : {cst.embed.vector_k}")
    print(f"Forecast horizon: {cst.forecast.forecast_horizon_days} days")

    total_steps = 5 + (1 if args.seed_kpi else 0)
    current_step = 0

    if args.seed_kpi:
        current_step += 1
        print_progress("Seeding KPI data", current_step, total_steps)
        seed_query = get_kpi_seeding_query(cfg)
        _ = measure_query_performance(
            client, seed_query, description="Seeding KPI data from thelook dataset"
        )
        print("KPI data seeded successfully!")

    # Dataset/state snapshot
    current_step += 1
    print_progress("Collecting dataset snapshot", current_step, total_steps)
    counts_sql = build_counts_query(cfg)
    counts_time, counts_bytes, counts_cache, counts_slot_ms, _ = (
        measure_query_performance(
            client,
            counts_sql,
            description="Computing table counts & index health",
        )
    )
    counts_row = list(client.query(counts_sql).result())[0]

    # Retrieval timing
    current_step += 1
    print_progress("Measuring retrieval latency", current_step, total_steps)
    context_query = build_context_query_with_fallback(cfg, cst, "@query_text")
    retrieval_time, retrieval_bytes, retrieval_cache, retrieval_slot_ms, _ = (
        measure_query_performance(
            client,
            context_query,
            params=[bigquery.ScalarQueryParameter("query_text", "STRING", args.query)],
            description="Retrieval (VECTOR_SEARCH + fallback)",
        )
    )

    # Question answering
    current_step += 1
    print_progress("Running question answering", current_step, total_steps)
    answer_query = synthesize_answer_sql(
        cfg,
        "answer_v1",
        "@query_text",
        context_query,
        cst.chunk.topk_context,
    )
    answer_time, answer_bytes, answer_cache, answer_slot_ms, answer_stmt = (
        measure_query_performance(
            client,
            answer_query,
            params=[bigquery.ScalarQueryParameter("query_text", "STRING", args.query)],
            description="Answer synthesis",
        )
    )

    # Forecasting
    current_step += 1
    print_progress("Running forecasting", current_step, total_steps)
    forecast_query = forecast_sql(cfg, cst, "@kpi")
    forecast_time, forecast_bytes, forecast_cache, forecast_slot_ms, forecast_stmt = (
        measure_query_performance(
            client,
            forecast_query,
            params=[bigquery.ScalarQueryParameter("kpi", "STRING", args.kpi)],
            description="Forecast",
        )
    )

    # Last answer preview and forecast summary
    current_step += 1
    print_progress("Summarizing outputs", current_step, total_steps)
    ans_prev_sql = build_answer_preview_query(cfg)
    ans_prev_rows = list(client.query(ans_prev_sql).result())
    ans_summary = ""
    ans_cits = 0
    ans_conf = None
    if ans_prev_rows:
        r = ans_prev_rows[0]
        ans_summary = (r["summary"] or "").replace("\n", " ")
        if len(ans_summary) > 160:
            ans_summary = ans_summary[:157] + "..."
        ans_cits = int(r["n_citations"]) if r["n_citations"] is not None else 0
        ans_conf = r["confidence"]

    fcast_sql = build_forecast_summary_query(cfg, "@kpi")
    fcast_rows = list(
        client.query(
            fcast_sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("kpi", "STRING", args.kpi)
                ]
            ),
        ).result()
    )
    fcast_p25 = fcast_p50 = fcast_p75 = None
    fcast_rows_prod = 0
    fcast_min_ts = fcast_max_ts = None
    if fcast_rows:
        rr = fcast_rows[0]
        fcast_rows_prod = int(rr["rows_prod"]) if rr["rows_prod"] is not None else 0
        fcast_min_ts = rr["min_ts"]
        fcast_max_ts = rr["max_ts"]
        fcast_p25 = rr["p25"]
        fcast_p50 = rr["p50"]
        fcast_p75 = rr["p75"]

    # Report
    print("\n" + "=" * 80)
    print("FULL BENCHMARK REPORT")
    print("=" * 80)

    print("\nSection: Environment")
    print(
        f"  Project/Region/Dataset : {cfg.project_id} / {cfg.region} / {cfg.dataset_id}"
    )
    print(f"  Connection             : {cfg.connection_id_short()}")
    print(
        f"  Models                 : text={cst.embed.text_model}, image={cst.embed.image_model}"
    )
    print(
        f"  Vector dims/k          : text={cst.embed.text_vector_dim}, image={cst.embed.image_vector_dim}, k={cst.embed.vector_k}"
    )
    print(f"  Forecast horizon       : {cst.forecast.forecast_horizon_days} days")

    print("\nSection: Dataset snapshot (rows)")
    print(f"  objects_images         : {counts_row['objects_images']}")
    print(f"  objects_pdfs           : {counts_row['objects_pdfs']}")
    print(f"  objects_screens        : {counts_row['objects_screens']}")
    print(f"  image_desc             : {counts_row['image_desc']}")
    print(f"  text_chunks            : {counts_row['text_chunks']}")
    print(f"  text_emb               : {counts_row['text_emb']}")
    print(f"  image_emb              : {counts_row['image_emb']}")
    print(f"  facts_generic          : {counts_row['facts_generic']}")
    print(f"  answers                : {counts_row['answers']}")
    print(f"  kpi_series             : {counts_row['kpi_series']}")
    print(f"  kpi_forecast           : {counts_row['kpi_forecast']}")
    print(f"  metrics_validators     : {counts_row['metrics_validators']}")
    print(f"  metrics_retrieval      : {counts_row['metrics_retrieval']}")
    print(
        f"  indexes                : text_idx={counts_row['has_text_idx']}, image_idx={counts_row['has_image_idx']}"
    )
    print(
        f"  snapshot query         : {counts_time:0.2f}s, {counts_bytes:0.2f} MB, cache_hit={counts_cache}"
    )

    print("\nSection: Timings & bytes")
    print(
        "  Retrieval              : {:6.2f}s | {:7.2f} MB | cache={} | slot_ms={}".format(
            retrieval_time, retrieval_bytes, retrieval_cache, int(retrieval_slot_ms)
        )
    )
    print(
        "  Answer synthesis       : {:6.2f}s | {:7.2f} MB | cache={} | slot_ms={} | stmt={}".format(
            answer_time, answer_bytes, answer_cache, int(answer_slot_ms), answer_stmt
        )
    )
    print(
        "  Forecast               : {:6.2f}s | {:7.2f} MB | cache={} | slot_ms={} | stmt={}".format(
            forecast_time,
            forecast_bytes,
            forecast_cache,
            int(forecast_slot_ms),
            forecast_stmt,
        )
    )

    print("\nSection: Latest answer")
    print(f"  Summary                : {ans_summary}")
    print(f"  Citations              : {ans_cits}")
    print(f"  Confidence             : {ans_conf}")

    print("\nSection: Forecast summary")
    print(f"  Rows (prod)            : {fcast_rows_prod}")
    print(f"  Range (prod)           : {fcast_min_ts} → {fcast_max_ts}")
    print(f"  Predicted p25/p50/p75  : {fcast_p25} / {fcast_p50} / {fcast_p75}")

    print("\nSection: Run parameters")
    print(f"  Query text             : {args.query}")
    print(f"  KPI id                 : {args.kpi}")

    print("\n" + "=" * 80)
    print("Benchmarking completed successfully!")

    return 0


if __name__ == "__main__":
    exit(run())
