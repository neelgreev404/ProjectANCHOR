"""
Project Anchor orchestrator: assembles SQL for each stage and yields them in order.

Uses config and constants, builds SQL via dedicated builder modules.
Ready for execution via notebook run_sql() or a CLI runner.
"""

from __future__ import annotations

from typing import List

from .core import Config, AllConstants
from .builders import (
    text_embeddings_sql,
    image_embeddings_sql,
    image_descriptions_sql,
    pdf_text_chunks_sql,
    screens_text_chunks_sql,
    images_text_chunks_sql,
    extract_facts_sql,
    bool_validator_sql,
    double_tolerance_validator_sql,
    brand_model_match_sql,
    table_vs_text_total_sql,
    retrieval_eval_sql,
    hybrid_score_sql,
    precompute_neighbors_bulk_sql,
    create_hybrid_view_sql,
    retrieval_context_sql,
    synthesize_answer_sql,
    forecast_sql,
    schemas_ddl_sql,
)


def _index_sql(cfg: Config, *, cst: AllConstants | None = None) -> str:
    """Create/refresh vector indexes when needed."""
    text_emb = cfg.table("text_emb")
    image_emb = cfg.table("image_emb")
    idx_text = cfg.table("text_emb_idx")
    idx_image = cfg.table("image_emb_idx")
    metrics_index_state = cfg.table("metrics_index_state")
    growth_threshold_pct = cst.embed.index_growth_thresh_pct if cst else 15
    min_rows = 1
    return (
        f"DECLARE has_text_nonempty BOOL DEFAULT FALSE;\n"
        f"DECLARE has_image_nonempty BOOL DEFAULT FALSE;\n"
        f"DECLARE has_text_idx BOOL DEFAULT FALSE;\n"
        f"DECLARE has_image_idx BOOL DEFAULT FALSE;\n"
        f"DECLARE text_rows INT64 DEFAULT 0;\n"
        f"DECLARE image_rows INT64 DEFAULT 0;\n"
        f"DECLARE prev_text_rows INT64 DEFAULT 0;\n"
        f"DECLARE prev_image_rows INT64 DEFAULT 0;\n"
        f"DECLARE grow_text_pct FLOAT64 DEFAULT 0.0;\n"
        f"DECLARE grow_image_pct FLOAT64 DEFAULT 0.0;\n"
        f"DECLARE min_rows INT64 DEFAULT {min_rows};\n"
        f"SET has_text_nonempty = ((SELECT COUNTIF(ARRAY_LENGTH(vector) > 0) FROM {text_emb}) > 0);\n"
        f"SET has_image_nonempty = ((SELECT COUNTIF(ARRAY_LENGTH(vector) > 0) FROM {image_emb}) > 0);\n"
        f"SET has_text_idx = (SELECT COUNT(*) > 0 FROM {cfg.table('INFORMATION_SCHEMA.VECTOR_INDEXES')} WHERE index_name = 'text_emb_idx');\n"
        f"SET has_image_idx = (SELECT COUNT(*) > 0 FROM {cfg.table('INFORMATION_SCHEMA.VECTOR_INDEXES')} WHERE index_name = 'image_emb_idx');\n"
        f"DELETE FROM {text_emb} WHERE vector IS NULL OR ARRAY_LENGTH(vector) = 0;\n"
        f"DELETE FROM {image_emb} WHERE vector IS NULL OR ARRAY_LENGTH(vector) = 0;\n"
        f"SET text_rows = (SELECT COUNT(*) FROM {text_emb});\n"
        f"SET image_rows = (SELECT COUNT(*) FROM {image_emb});\n"
        f"IF has_text_nonempty AND NOT has_text_idx AND text_rows >= min_rows THEN\n"
        f"  CREATE VECTOR INDEX {idx_text} ON {text_emb} (vector) OPTIONS (index_type = 'IVF', distance_type = 'COSINE');\n"
        f"END IF;\n"
        f"IF has_image_nonempty AND NOT has_image_idx AND image_rows >= min_rows THEN\n"
        f"  CREATE VECTOR INDEX {idx_image} ON {image_emb} (vector) OPTIONS (index_type = 'IVF', distance_type = 'COSINE');\n"
        f"END IF;\n"
        f"SET prev_text_rows = COALESCE((SELECT last_row_count FROM {metrics_index_state} WHERE index_name = 'text_emb_idx'), 0);\n"
        f"SET prev_image_rows = COALESCE((SELECT last_row_count FROM {metrics_index_state} WHERE index_name = 'image_emb_idx'), 0);\n"
        f"SET grow_text_pct = SAFE_DIVIDE(100.0 * (text_rows - prev_text_rows), GREATEST(prev_text_rows, 1));\n"
        f"SET grow_image_pct = SAFE_DIVIDE(100.0 * (image_rows - prev_image_rows), GREATEST(prev_image_rows, 1));\n"
        f"IF has_text_nonempty AND has_text_idx AND text_rows >= min_rows AND grow_text_pct > (SELECT index_growth_thresh_pct FROM UNNEST([STRUCT({growth_threshold_pct} AS index_growth_thresh_pct)])) THEN\n"
        f"  DROP INDEX {idx_text};\n"
        f"  CREATE VECTOR INDEX {idx_text} ON {text_emb} (vector) OPTIONS (index_type = 'IVF', distance_type = 'COSINE');\n"
        f"END IF;\n"
        f"IF has_image_nonempty AND has_image_idx AND image_rows >= min_rows AND grow_image_pct > (SELECT index_growth_thresh_pct FROM UNNEST([STRUCT({growth_threshold_pct} AS index_growth_thresh_pct)])) THEN\n"
        f"  DROP INDEX {idx_image};\n"
        f"  CREATE VECTOR INDEX {idx_image} ON {image_emb} (vector) OPTIONS (index_type = 'IVF', distance_type = 'COSINE');\n"
        f"END IF;\n"
        f"MERGE {metrics_index_state} dst USING (SELECT 'text_emb_idx' AS index_name, text_rows AS last_row_count) src ON dst.index_name = src.index_name\n"
        f"WHEN MATCHED THEN UPDATE SET last_row_count = src.last_row_count, updated_ts = CURRENT_TIMESTAMP()\n"
        f"WHEN NOT MATCHED THEN INSERT (index_name, last_row_count) VALUES (src.index_name, src.last_row_count);\n"
        f"MERGE {metrics_index_state} dst USING (SELECT 'image_emb_idx' AS index_name, image_rows AS last_row_count) src ON dst.index_name = src.index_name\n"
        f"WHEN MATCHED THEN UPDATE SET last_row_count = src.last_row_count, updated_ts = CURRENT_TIMESTAMP()\n"
        f"WHEN NOT MATCHED THEN INSERT (index_name, last_row_count) VALUES (src.index_name, src.last_row_count);"
    )


def _index_health_gate_sql(cfg: Config) -> str:
    """Create indexes only when vectors exist."""
    return (
        f"DECLARE has_text_vec BOOL DEFAULT (SELECT COUNTIF(ARRAY_LENGTH(vector) > 0) > 0 FROM {cfg.table('text_emb')});\n"
        f"DECLARE has_image_vec BOOL DEFAULT (SELECT COUNTIF(ARRAY_LENGTH(vector) > 0) > 0 FROM {cfg.table('image_emb')});\n"
        f"DECLARE text_rows INT64 DEFAULT (SELECT COUNT(*) FROM {cfg.table('text_emb')});\n"
        f"DECLARE image_rows INT64 DEFAULT (SELECT COUNT(*) FROM {cfg.table('image_emb')});\n"
        f"DECLARE min_rows INT64 DEFAULT 1;\n"
        f"IF has_text_vec AND text_rows >= min_rows THEN\n"
        f"  CREATE OR REPLACE VECTOR INDEX {cfg.table('text_emb_idx')} ON {cfg.table('text_emb')} (vector) OPTIONS (index_type = 'IVF', distance_type = 'COSINE');\n"
        f"END IF;\n"
        f"IF has_image_vec AND image_rows >= min_rows THEN\n"
        f"  CREATE OR REPLACE VECTOR INDEX {cfg.table('image_emb_idx')} ON {cfg.table('image_emb')} (vector) OPTIONS (index_type = 'IVF', distance_type = 'COSINE');\n"
        f"END IF;"
    )


def _purge_transient_sql(cfg: Config) -> str:
    """Clean up old transient data."""
    edges = cfg.table("edges_semantic")
    neighbors = cfg.table("precomputed_neighbors")
    return (
        f"DELETE FROM {edges} WHERE TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), created_ts, MINUTE) > {AllConstants().eval.cache_ttl_min};\n"
        f"DELETE FROM {neighbors} WHERE TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), created_ts, MINUTE) > {AllConstants().eval.cache_ttl_min};"
    )


def _seed_eval_sql(cfg: Config, cst: AllConstants) -> str:
    """Set up basic eval data."""
    text_model = cst.embed.text_model
    hot_q = cfg.table("hot_queries")
    gt = cfg.table("retrieval_gt")
    text_emb = cfg.table("text_emb")
    return (
        f"DECLARE has_hot BOOL DEFAULT FALSE;\n"
        f"DECLARE has_gt BOOL DEFAULT FALSE;\n"
        f"SET has_hot = (SELECT COUNT(*) > 0 FROM {hot_q});\n"
        f"SET has_gt = (SELECT COUNT(*) > 0 FROM {gt});\n"
        f"IF NOT has_hot THEN\n"
        f"  INSERT INTO {hot_q} (query_id, query_text, embedding)\n"
        f"  SELECT 'q_demo', 'demo query', (\n"
        f"    SELECT ml_generate_embedding_result\n"
        f"    FROM ML.GENERATE_EMBEDDING(\n"
        f"      MODEL `{text_model}`,\n"
        f"      (SELECT 'demo query' AS content)\n"
        f"    )\n"
        f"  );\n"
        f"END IF;\n"
        f"IF NOT has_gt THEN\n"
        f"  IF (SELECT COUNT(*) FROM {text_emb}) > 0 THEN\n"
        f"    INSERT INTO {gt} (query_id, item_id)\n"
        f"    SELECT 'q_demo', (SELECT chunk_id FROM {text_emb} LIMIT 1);\n"
        f"  END IF;\n"
        f"END IF;"
    )


def _seed_objects_images_sql(cfg: Config) -> str:
    """Add a demo image if none exist."""
    objects = cfg.table("objects_images")
    uri = "gs://cloud-samples-data/bigquery/tutorials/cymbal-pets/images/cat.jpg"
    return (
        f"DECLARE has_objs BOOL DEFAULT (SELECT COUNT(*) > 0 FROM {objects});\n"
        f"IF NOT has_objs THEN\n"
        f"  INSERT INTO {objects} (object_uri, media_type, object_ref) VALUES ('{uri}', 'image/jpeg', NULL);\n"
        f"END IF;"
    )


def _seed_text_chunks_from_captions_sql(cfg: Config) -> str:
    """Create text chunks from image captions."""
    chunks = cfg.table("text_chunks")
    image_desc = cfg.table("image_desc")
    return (
        f"MERGE {chunks} AS dst\n"
        f"USING (\n"
        f"  SELECT TO_HEX(SHA256(CONCAT(source_uri, '#cap'))) AS chunk_id,\n"
        f"         source_uri, caption AS text, CAST(NULL AS STRING) AS lang, STRUCT(0 AS offset_start, LENGTH(caption) AS offset_end) AS span\n"
        f"  FROM {image_desc} WHERE caption IS NOT NULL\n"
        f") AS src\n"
        f"ON dst.chunk_id = src.chunk_id\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (chunk_id, source_uri, text, lang, span) VALUES (src.chunk_id, src.source_uri, src.text, src.lang, src.span);"
    )


def _ensure_one_chunk_from_captions_sql(cfg: Config) -> str:
    """Ensure we have at least one text chunk."""
    chunks = cfg.table("text_chunks")
    image_desc = cfg.table("image_desc")
    return (
        f"MERGE {chunks} AS dst\n"
        f"USING (\n"
        f"  SELECT TO_HEX(SHA256(CONCAT(source_uri, '#cap'))) AS chunk_id,\n"
        f"         source_uri, caption AS text, CAST(NULL AS STRING) AS lang, STRUCT(0 AS offset_start, LENGTH(caption) AS offset_end) AS span\n"
        f"  FROM {image_desc}\n"
        f"  WHERE caption IS NOT NULL\n"
        f"  LIMIT 1\n"
        f") AS src\n"
        f"ON dst.chunk_id = src.chunk_id\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (chunk_id, source_uri, text, lang, span) VALUES (src.chunk_id, src.source_uri, src.text, src.lang, src.span);"
    )


def _semantic_edges_sql(cfg: Config, cst: AllConstants) -> str:
    """Build semantic edges using vector search."""
    text_emb = cfg.table("text_emb")
    k = cst.embed.vector_k
    edges = cfg.table("edges_semantic")
    return (
        f"DECLARE v ARRAY<FLOAT64>;\n"
        f"DECLARE v_q STRING;\n"
        f"FOR rec IN (SELECT query_id, embedding FROM {cfg.table('hot_queries')}) DO\n"
        f"  SET v = rec.embedding;\n"
        f"  SET v_q = rec.query_id;\n"
        f"  MERGE {edges} AS dst\n"
        f"  USING (\n"
        f"    WITH vs_raw AS (\n"
        f"      SELECT TO_JSON(vs) AS vs_json, vs.distance\n"
        f"      FROM VECTOR_SEARCH(TABLE {text_emb}, 'vector', (SELECT v AS vector), top_k => {k}) AS vs\n"
        f"    ), ids AS (\n"
        f"      SELECT COALESCE(JSON_VALUE(vs_json, '$.id'), JSON_VALUE(vs_json, '$.row_id')) AS rid, distance\n"
        f"      FROM vs_raw\n"
        f"      WHERE COALESCE(JSON_VALUE(vs_json, '$.id'), JSON_VALUE(vs_json, '$.row_id')) IS NOT NULL\n"
        f"    )\n"
        f"    SELECT v_q AS left_id, ids.rid AS right_id, 'text' AS left_type, 'text' AS right_type, 1.0 - ids.distance AS similarity, TRUE AS constraints_ok, CAST(NULL AS STRING) AS constraint_notes\n"
        f"    FROM ids\n"
        f"  ) AS src\n"
        f"  ON dst.left_id = src.left_id AND dst.right_id = src.right_id AND dst.left_type = src.left_type AND dst.right_type = src.right_type\n"
        f"  WHEN MATCHED THEN UPDATE SET similarity = src.similarity, constraints_ok = src.constraints_ok, constraint_notes = src.constraint_notes\n"
        f"  WHEN NOT MATCHED THEN INSERT (edge_id, left_id, right_id, left_type, right_type, similarity, constraints_ok, constraint_notes)\n"
        f"    VALUES (GENERATE_UUID(), src.left_id, src.right_id, src.left_type, src.right_type, src.similarity, src.constraints_ok, src.constraint_notes);\n"
        f"END FOR;"
    )


def _seed_holdout_sql(cfg: Config, cst: AllConstants) -> str:
    """Create holdout data for evaluation."""
    text_model = cst.embed.text_model
    hot_q = cfg.table("hot_queries")
    gt = cfg.table("retrieval_gt")
    chunks = cfg.table("text_chunks")
    holdout = cst.eval.holdout_size
    return (
        f"DECLARE has_hot BOOL DEFAULT (SELECT COUNT(*) > 0 FROM {hot_q});\n"
        f"DECLARE has_gt BOOL DEFAULT (SELECT COUNT(*) > 0 FROM {gt});\n"
        f"IF NOT has_hot THEN\n"
        f"  INSERT INTO {hot_q} (query_id, query_text, embedding)\n"
        f"  SELECT CONCAT('hq_', CAST(ROW_NUMBER() OVER (ORDER BY c.chunk_id) AS STRING)) AS query_id,\n"
        f"         c.text AS query_text,\n"
        f"         COALESCE(e.vector, (SELECT ml_generate_embedding_result FROM ML.GENERATE_EMBEDDING( MODEL `{text_model}`, (SELECT c.text AS content)))) AS embedding\n"
        f"  FROM (SELECT chunk_id, text FROM {chunks} LIMIT {holdout}) c\n"
        f"  LEFT JOIN {cfg.table('text_emb')} e ON e.chunk_id = c.chunk_id;\n"
        f"END IF;\n"
        f"IF NOT has_gt THEN\n"
        f"  INSERT INTO {gt} (query_id, item_id)\n"
        f"  SELECT CONCAT('hq_', CAST(ROW_NUMBER() OVER (ORDER BY chunk_id) AS STRING)), chunk_id\n"
        f"  FROM (SELECT chunk_id FROM {chunks} LIMIT {holdout});\n"
        f"END IF;"
    )


def _kpi_series_from_facts_sql(cfg: Config) -> str:
    """Build KPI time series from facts."""
    facts = cfg.table("facts_generic")
    series = cfg.table("kpi_series")
    return (
        f"MERGE {series} AS dst\n"
        f"USING (\n"
        f"  SELECT 'demo_kpi' AS kpi_id,\n"
        f"         IFNULL(object_date, DATE(CURRENT_TIMESTAMP())) AS ts,\n"
        f"         SUM(IFNULL(object_number, 0.0)) AS value,\n"
        f"         ARRAY<STRING>[] AS tags\n"
        f"  FROM {facts}\n"
        f"  GROUP BY ts\n"
        f") AS src\n"
        f"ON dst.kpi_id = src.kpi_id AND dst.ts = src.ts\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (kpi_id, ts, value, tags) VALUES (src.kpi_id, src.ts, src.value, src.tags);"
    )


def _seed_kpi_orders_from_thelook_sql(cfg: Config) -> str:
    series = cfg.table("kpi_series")
    return (
        f"MERGE {series} AS dst\n"
        f"USING (\n"
        f"  SELECT 'orders' AS kpi_id, DATE(created_at) AS ts, COUNT(*) AS value, ARRAY<STRING>[] AS tags\n"
        f"  FROM `bigquery-public-data.thelook_ecommerce.orders`\n"
        f"  WHERE created_at IS NOT NULL AND DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY)\n"
        f"  GROUP BY ts\n"
        f") AS src\n"
        f"ON dst.kpi_id = src.kpi_id AND dst.ts = src.ts\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (kpi_id, ts, value, tags) VALUES (src.kpi_id, src.ts, src.value, src.tags);"
    )


def assemble_pipeline_sql(cfg: Config, cst: AllConstants) -> List[str]:
    """Build the complete SQL pipeline in the right order."""
    statements: List[str] = []
    # Start with schema setup
    statements.append(schemas_ddl_sql(cfg))
    # Add demo data if needed
    statements.append(_seed_objects_images_sql(cfg))

    if not cfg.quickstart:
        statements.append(image_descriptions_sql(cfg))
        statements.append(_seed_text_chunks_from_captions_sql(cfg))
        statements.append(_ensure_one_chunk_from_captions_sql(cfg))

    # Generate embeddings
    if not cfg.quickstart:
        statements.append(pdf_text_chunks_sql(cfg))
        statements.append(screens_text_chunks_sql(cfg))
        statements.append(images_text_chunks_sql(cfg))
        statements.append(text_embeddings_sql(cfg, cst))
        statements.append(image_embeddings_sql(cfg, cst))

    # Create indexes
    statements.append(_index_sql(cfg, cst=cst))

    # Extract facts
    if not cfg.quickstart:
        statements.append(extract_facts_sql(cfg, cst, prompt_id="facts_v1"))

    # Set up evaluation data
    if not cfg.quickstart:
        statements.append(_seed_eval_sql(cfg, cst))
        statements.append(_seed_holdout_sql(cfg, cst))

    # Run validators
    if not cfg.quickstart:
        sample = min(1.0, cst.eval.max_validation_sample_frac)
        statements.append(
            bool_validator_sql(
                cfg,
                "rule_color_match",
                "image_desc",
                "image_id",
                "STRUCT(caption AS text)",
                sample_frac=sample,
            )
        )
        statements.append(
            double_tolerance_validator_sql(
                cfg,
                cst,
                "rule_total",
                "facts_generic",
                "fact_id",
                "STRUCT(object_value AS text)",
                "object_number",
                0.02,
                sample_frac=sample,
            )
        )
        statements.append(brand_model_match_sql(cfg))
        statements.append(table_vs_text_total_sql(cfg))

    # Build retrieval system
    if not cfg.quickstart:
        statements.append(_purge_transient_sql(cfg))
        statements.append(precompute_neighbors_bulk_sql(cfg, cst))
        statements.append(_semantic_edges_sql(cfg, cst))

    # Evaluate retrieval
    if not cfg.quickstart:
        statements.append(retrieval_eval_sql(cfg, cst))

    # Demo hybrid ranking
    if not cfg.quickstart:
        hscore = hybrid_score_sql(
            cst.eval.hybrid_alpha,
            cst.eval.hybrid_beta,
            cst.eval.hybrid_gamma,
            cst.eval.hybrid_delta,
        )
        statements.append(
            f"SELECT *, {hscore}, 0.0 AS recency_score, 0.0 AS validator_pass_rate FROM {cfg.table('edges_semantic')} ORDER BY hybrid_score DESC LIMIT 10"
        )
        statements.append(create_hybrid_view_sql(cfg, cst))

    # Answer generation
    _k = cst.chunk.topk_context
    _primary_ctx = retrieval_context_sql(cfg, cst, "@query_text", _k)
    ctx_sql = (
        f"WITH ctx_primary AS (\n"
        f"  {_primary_ctx}\n"
        f"), ctx_backup AS (\n"
        f"  SELECT source_uri, caption AS text\n"
        f"  FROM {cfg.table('image_desc')}\n"
        f"  WHERE caption IS NOT NULL\n"
        f"  ORDER BY created_ts DESC\n"
        f"  LIMIT {_k}\n"
        f"), ctx_union AS (\n"
        f"  SELECT * FROM ctx_primary\n"
        f"  UNION ALL\n"
        f"  SELECT * FROM ctx_backup\n"
        f")\n"
        f"SELECT source_uri, text FROM ctx_union LIMIT {_k}"
    )
    context_limit = 1 if cfg.quickstart else _k
    statements.append(
        synthesize_answer_sql(cfg, "answer_v1", "@query_text", ctx_sql, context_limit)
    )

    # Build KPI data
    if not cfg.quickstart:
        statements.append(_kpi_series_from_facts_sql(cfg))
        statements.append(_seed_kpi_orders_from_thelook_sql(cfg))

    # Generate forecasts
    if not cfg.quickstart:
        statements.append(forecast_sql(cfg, cst, "@kpi"))

    # Create views
    if not cfg.quickstart:
        statements.append(
            f"CREATE OR REPLACE VIEW `{cfg.project_id}.{cfg.dataset_id}.v_kpi_cards` AS\n"
            f"SELECT kpi_id, ts, predicted, lower, upper FROM {cfg.table('kpi_forecast')} WHERE backtest = FALSE ORDER BY ts DESC LIMIT 100;"
        )
    return statements


def main() -> None:
    cfg = Config.from_env()
    cst = AllConstants()
    cst.validate()
    for i, sql in enumerate(assemble_pipeline_sql(cfg, cst), start=1):
        print(f"-- Statement {i} --\n{sql}\n")


def smoke() -> int:
    cfg = Config.from_env()
    cst = AllConstants()
    assert assemble_pipeline_sql(cfg, cst), "Orchestrator produced no statements"
    return 0


if __name__ == "__main__":
    main()
