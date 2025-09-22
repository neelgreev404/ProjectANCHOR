"""
Project Anchor SQL builders.

- Answer synthesis
- Embeddings and extraction (text/PDF/screenshots/images)
- Facts extraction
- Retrieval, hybrid scoring, and evaluation
- Validators (BOOL/DOUBLE)
- Forecasting
- Canonical DDL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import os
import json
from pathlib import Path

from .core import Config, AllConstants


# model params for AI.GENERATE (short + deterministic keeps latency/cost predictable)
MODEL_PARAMS_JSON = '{"generation_config":{"max_output_tokens":64,"temperature":0}}'

# BigQuery output_schema must be a type-spec string (not JSON)
ANSWER_SCHEMA_SPEC = (
    "summary STRING, citations ARRAY<STRING>, confidence FLOAT64, "
    "validators ARRAY<STRUCT<name STRING, result BOOL, note STRING>>"
)


def _load_prompt(cfg: Config, name: str, default_text: str) -> str:
    try:
        pdir = Path(cfg.prompts_dir or "prompts")
        # Support either .json with {"instruction": "..."} or .txt with raw instruction
        json_path = pdir / f"{name}.json"
        txt_path = pdir / f"{name}.txt"
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("instruction"):
                return str(data["instruction"])
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return default_text


def _prompt_answer_instruction(cfg: Config) -> str:
    return _load_prompt(
        cfg,
        os.getenv("ANSWER_PROMPT_ID", "answer_v1"),
        "Given the query and contexts, produce a short, cited answer.",
    )


def _prompt_facts_instruction(cfg: Config) -> str:
    return _load_prompt(
        cfg,
        os.getenv("FACTS_PROMPT_ID", "facts_v1"),
        "Extract facts: entity, predicate, values (text/number/date) and confidence from the snippet",
    )


def _prompt_image_caption(cfg: Config) -> str:
    return _load_prompt(cfg, "image_caption_v1", "Describe the image")


def _prompt_text_extract(cfg: Config) -> str:
    return _load_prompt(cfg, "text_extract_v1", "Extract text:")


def synthesize_answer_sql(
    cfg: Config,
    prompt_id: str,
    query_param: str,
    context_source_sql: str,
    context_limit: int,
) -> str:
    """Compose SQL to synthesize a cited answer from a small, relevant context."""
    answers = cfg.table("answers")
    prompt_log = cfg.table("prompt_log")
    conn = cfg.connection_id_short()
    conn_named = f", connection_id => '{conn}'" if conn else ""

    # Deterministic A/B split for prompt_version using FARM_FINGERPRINT of query text
    ab_split_frac_str = "0.5"
    try:
        from .core import AllConstants as _All  # type: ignore

        ab_split_frac_str = str(_All().eval.prompt_ab_split_frac)
    except Exception:
        # Fall back to 0.5 if constants are unavailable
        pass

    # Optional validator aggregation CTE (skipped in quickstart to save latency)
    v_cte = (
        (
            f", v AS (\n"
            f"  SELECT ARRAY_AGG(STRUCT(name AS name, result AS result, note AS note)) AS validators,\n"
            f"         TO_JSON_STRING(ARRAY_AGG(STRUCT(name, result, note))) AS validators_summary\n"
            f"  FROM (\n"
            f"    SELECT rule_id AS name, LOGICAL_AND(result_bool) AS result, ANY_VALUE(note) AS note\n"
            f"    FROM {cfg.table('metrics_validators')}\n"
            f"    GROUP BY rule_id\n"
            f"  )\n"
            f" )\n"
        )
        if not cfg.quickstart
        else ""
    )

    # Optional field attached to the input struct when validators are materialized
    extra_summary_field = (
        ", (SELECT validators_summary FROM v) AS validators_summary"
        if not cfg.quickstart
        else ""
    )

    answer_instruction = _prompt_answer_instruction(cfg)

    # Build sanitized snippets and carry forward allowed_uris for strict citation filtering
    return (
        f"INSERT INTO {answers} (answer_id, query, summary, citations, validators, confidence)\n"
        f"WITH ctx AS (\n"
        f"  SELECT * FROM (\n"
        f"    {context_source_sql}\n"
        f"  )\n"
        f"  LIMIT {context_limit}\n"
        f"), agg AS (\n"
        f"  SELECT\n"
        f"    CAST({query_param} AS STRING) AS query,\n"
        f"    STRING_AGG(SUBSTR(REPLACE(REPLACE(ctx.text, '\\n', ' '), '\\r', ' '), 1, 240), '\\n- ') AS context_text,\n"
        f"    ARRAY_AGG(DISTINCT ctx.source_uri) AS allowed_uris\n"
        f"  FROM ctx\n"
        f" )\n" + v_cte + f", gen AS (\n"
        f"  SELECT AI.GENERATE(\n"
        f"    STRUCT('{answer_instruction}' AS instruction, CAST(a.query AS STRING) AS query, a.context_text AS context, CONCAT('Allowed sources: ', ARRAY_TO_STRING(a.allowed_uris, ', ')) AS allowed_sources{extra_summary_field}),\n"
        f"    output_schema => '{ANSWER_SCHEMA_SPEC}'{conn_named},\n"
        f"    model_params => JSON '{MODEL_PARAMS_JSON}'\n"
        f"  ) AS out, a.allowed_uris AS allowed_uris\n"
        f"  FROM agg a\n"
        f")\n"
        f"SELECT\n"
        f"  GENERATE_UUID() AS answer_id,\n"
        f"  CAST({query_param} AS STRING) AS query,\n"
        f"  COALESCE(out.summary, '') AS summary,\n"
        f"  (SELECT ARRAY( SELECT uri FROM UNNEST(out.citations) AS uri WHERE uri IN UNNEST(allowed_uris) )) AS citations,\n"
        f"  {('(SELECT validators FROM v)' if not cfg.quickstart else '(SELECT ARRAY(SELECT AS STRUCT name, result, note FROM UNNEST(out.validators)))')} AS validators,\n"
        f"  GREATEST(0.0, LEAST(1.0, out.confidence)) AS confidence\n"
        f"FROM gen;\n"
        f"INSERT INTO {prompt_log} (call_id, prompt_id, prompt_version, model_id, input_hash)\n"
        f"SELECT GENERATE_UUID(), '{prompt_id}', CASE WHEN MOD(ABS(FARM_FINGERPRINT(CAST({query_param} AS STRING))), 1000) < CAST(ROUND({ab_split_frac_str}*1000) AS INT64) THEN 'answer_v1' ELSE 'answer_v2' END, 'AI.GENERATE', TO_HEX(SHA256(CAST({query_param} AS STRING)));"
    )


def text_embeddings_sql(cfg: Config, cst: AllConstants) -> str:
    """Idempotent MERGE to populate `text_emb` via ML.GENERATE_EMBEDDING.

    Optimizations:
    - Only embed rows that are missing in `text_emb`
    - Optional cap via EMBED_MAX_ROWS_TEXT env (default: no cap)
    """
    text_chunks = cfg.table("text_chunks")
    text_emb = cfg.table("text_emb")
    model = cst.embed.text_model
    vec_dim = cst.embed.text_vector_dim
    cap_env = os.getenv("EMBED_MAX_ROWS_TEXT", "").strip()
    cap_clause = (
        f" LIMIT {int(cap_env)}" if cap_env.isdigit() and int(cap_env) > 0 else ""
    )
    return (
        f"MERGE {text_emb} AS dst\n"
        f"USING (\n"
        f"  WITH pending AS (\n"
        f"    SELECT c.chunk_id, c.text AS content\n"
        f"    FROM {text_chunks} c\n"
        f"    LEFT JOIN {text_emb} e USING (chunk_id)\n"
        f"    WHERE e.chunk_id IS NULL{cap_clause}\n"
        f"  )\n"
        f"  SELECT\n"
        f"    chunk_id, '{model}' AS model_id,\n"
        f"    ml_generate_embedding_result AS vector,\n"
        f"    {vec_dim} AS vector_dim\n"
        f"  FROM ML.GENERATE_EMBEDDING(\n"
        f"    MODEL `{model}`,\n"
        f"    (SELECT chunk_id, content FROM pending),\n"
        f"    STRUCT(TRUE AS flatten_json_output)\n"
        f"  )\n"
        f") AS src\n"
        f"ON dst.chunk_id = src.chunk_id\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (chunk_id, model_id, vector, vector_dim) VALUES (src.chunk_id, src.model_id, src.vector, src.vector_dim);"
    )


def image_descriptions_sql(cfg: Config) -> str:
    """Generate image captions and a few attributes using AI.GENERATE (idempotent)."""
    objects_view = cfg.table("v_objects_images")
    image_desc = cfg.table("image_desc")
    conn_short = cfg.connection_id_short()
    caption_instr = _prompt_image_caption(cfg)

    return (
        f"MERGE {image_desc} AS dst\n"
        f"USING (\n"
        f"  WITH pending AS (\n"
        f"    SELECT o.object_uri, o.object_ref\n"
        f"    FROM {objects_view} o\n"
        f"    LEFT JOIN {image_desc} d ON TO_HEX(SHA256(o.object_uri)) = d.image_id\n"
        f"    WHERE d.image_id IS NULL\n"
        f"  ), gen AS (\n"
        f"    SELECT\n"
        f"      TO_HEX(SHA256(p.object_uri)) AS image_id,\n"
        f"      p.object_uri AS source_uri,\n"
        f"      (AI.GENERATE(\n"
        f"         STRUCT('{caption_instr}' AS instruction, OBJ.GET_ACCESS_URL(p.object_ref, 'r') AS ref),\n"
        f"         connection_id => '{conn_short}',\n"
        f"         output_schema => 'caption STRING, dominant_colors ARRAY<STRING>, width_px INT64, height_px INT64'\n"
        f"       )) AS out\n"
        f"    FROM pending p\n"
        f"  )\n"
        f"  SELECT image_id, source_uri, out.caption AS caption, out.dominant_colors AS dominant_colors, out.width_px AS width_px, out.height_px AS height_px\n"
        f"  FROM gen\n"
        f") AS src\n"
        f"ON dst.image_id = src.image_id\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (image_id, source_uri, caption, dominant_colors, width_px, height_px)\n"
        f"  VALUES (src.image_id, src.source_uri, src.caption, src.dominant_colors, src.width_px, src.height_px);\n"
        f"INSERT INTO {cfg.table('prompt_log')} (call_id, prompt_id, prompt_version, model_id, input_hash)\n"
        f"SELECT GENERATE_UUID(), 'image_desc_v1', 'image_desc_v1', 'AI.GENERATE', TO_HEX(SHA256(o.object_uri))\n"
        f"FROM {objects_view} AS o\n"
        f"LEFT JOIN {image_desc} d ON TO_HEX(SHA256(o.object_uri)) = d.image_id\n"
        f"WHERE d.image_id IS NULL;"
    )


def image_embeddings_sql(cfg: Config, cst: AllConstants) -> str:
    """Embed images using captions from image_desc, only for missing rows.

    Optimizations:
    - Only embed rows that are missing in `image_emb`
    - Optional cap via EMBED_MAX_ROWS_IMAGE env (default: no cap)
    """
    image_desc = cfg.table("image_desc")
    image_emb = cfg.table("image_emb")
    model = cst.embed.image_model
    vec_dim = cst.embed.image_vector_dim
    cap_env = os.getenv("EMBED_MAX_ROWS_IMAGE", "").strip()
    cap_clause = (
        f" LIMIT {int(cap_env)}" if cap_env.isdigit() and int(cap_env) > 0 else ""
    )
    return (
        f"MERGE {image_emb} AS dst\n"
        f"USING (\n"
        f"  WITH pending AS (\n"
        f"    SELECT d.image_id, d.caption AS content\n"
        f"    FROM {image_desc} d\n"
        f"    LEFT JOIN {image_emb} e USING (image_id)\n"
        f"    WHERE d.caption IS NOT NULL AND e.image_id IS NULL{cap_clause}\n"
        f"  )\n"
        f"  SELECT\n"
        f"    image_id, '{model}' AS model_id,\n"
        f"    ml_generate_embedding_result AS vector,\n"
        f"    {vec_dim} AS vector_dim\n"
        f"  FROM ML.GENERATE_EMBEDDING(\n"
        f"    MODEL `{model}`,\n"
        f"    (SELECT image_id, content FROM pending),\n"
        f"    STRUCT(TRUE AS flatten_json_output)\n"
        f"  )\n"
        f") AS src\n"
        f"ON dst.image_id = src.image_id\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (image_id, model_id, vector, vector_dim) VALUES (src.image_id, src.model_id, src.vector, src.vector_dim);"
    )


def pdf_text_chunks_sql(cfg: Config) -> str:
    """Extract text from PDFs into `text_chunks` using AI.GENERATE."""
    pdfs_view = cfg.table("v_objects_pdfs")
    chunks = cfg.table("text_chunks")
    conn_short = cfg.connection_id_short()
    extract_instr = _prompt_text_extract(cfg)
    return (
        f"MERGE {chunks} AS dst\n"
        f"USING (\n"
        f"  SELECT\n"
        f"    chunk_id,\n"
        f"    source_uri,\n"
        f"    out.text AS text,\n"
        f"    CAST(NULL AS STRING) AS lang,\n"
        f"    STRUCT(0 AS offset_start, LENGTH(out.text) AS offset_end) AS span\n"
        f"  FROM (\n"
        f"    SELECT\n"
        f"      TO_HEX(SHA256(CONCAT(object_uri, '#pdf'))) AS chunk_id,\n"
        f"      object_uri AS source_uri,\n"
        f"      AI.GENERATE(\n"
        f"        STRUCT('{extract_instr}' AS instruction, OBJ.GET_ACCESS_URL(object_ref, 'r') AS ref),\n"
        f"        connection_id => '{conn_short}',\n"
        f"        output_schema => 'text STRING'\n"
        f"      ) AS out\n"
        f"    FROM {pdfs_view}\n"
        f"  )\n"
        f"  WHERE out.text is NOT NULL AND LENGTH(out.text) > 0\n"
        f") AS src\n"
        f"ON dst.chunk_id = src.chunk_id\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (chunk_id, source_uri, text, lang, span) VALUES (src.chunk_id, src.source_uri, src.text, src.lang, src.span);\n"
        f"INSERT INTO {cfg.table('prompt_log')} (call_id, prompt_id, prompt_version, model_id, input_hash)\n"
        f"SELECT GENERATE_UUID(), 'text_extract_v1', 'text_extract_v1', 'AI.GENERATE', TO_HEX(SHA256(o.object_uri))\n"
        f"FROM {pdfs_view} AS o\n"
        f"LEFT JOIN {chunks} d ON TO_HEX(SHA256(CONCAT(o.object_uri, '#pdf'))) = d.chunk_id\n"
        f"WHERE d.chunk_id IS NULL;"
    )


def screens_text_chunks_sql(cfg: Config) -> str:
    """Extract text from screenshots into `text_chunks` using AI.GENERATE."""
    screens_view = cfg.table("v_objects_screens")
    chunks = cfg.table("text_chunks")
    conn_short = cfg.connection_id_short()
    extract_instr = _prompt_text_extract(cfg)
    return (
        f"MERGE {chunks} AS dst\n"
        f"USING (\n"
        f"  SELECT\n"
        f"    chunk_id,\n"
        f"    source_uri,\n"
        f"    out.text AS text,\n"
        f"    CAST(NULL AS STRING) AS lang,\n"
        f"    STRUCT(0 AS offset_start, LENGTH(out.text) AS offset_end) AS span\n"
        f"  FROM (\n"
        f"    SELECT\n"
        f"      TO_HEX(SHA256(CONCAT(object_uri, '#screen'))) AS chunk_id,\n"
        f"      object_uri AS source_uri,\n"
        f"      AI.GENERATE(\n"
        f"        STRUCT('{extract_instr}' AS instruction, OBJ.GET_ACCESS_URL(object_ref, 'r') AS ref),\n"
        f"        connection_id => '{conn_short}',\n"
        f"        output_schema => 'text STRING'\n"
        f"      ) AS out\n"
        f"    FROM {screens_view}\n"
        f"  )\n"
        f"  WHERE out.text IS NOT NULL AND LENGTH(out.text) > 0\n"
        f") AS src\n"
        f"ON dst.chunk_id = src.chunk_id\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (chunk_id, source_uri, text, lang, span) VALUES (src.chunk_id, src.source_uri, src.text, src.lang, src.span);\n"
        f"INSERT INTO {cfg.table('prompt_log')} (call_id, prompt_id, prompt_version, model_id, input_hash)\n"
        f"SELECT GENERATE_UUID(), 'text_extract_v1', 'text_extract_v1', 'AI.GENERATE', TO_HEX(SHA256(o.object_uri))\n"
        f"FROM {screens_view} AS o\n"
        f"LEFT JOIN {chunks} d ON TO_HEX(SHA256(CONCAT(o.object_uri, '#screen'))) = d.chunk_id\n"
        f"WHERE d.chunk_id IS NULL;"
    )


def images_text_chunks_sql(cfg: Config) -> str:
    """Extract text from images into `text_chunks` using AI.GENERATE."""
    images_view = cfg.table("v_objects_images")
    chunks = cfg.table("text_chunks")
    conn_short = cfg.connection_id_short()
    extract_instr = _prompt_text_extract(cfg)
    return (
        f"MERGE {chunks} AS dst\n"
        f"USING (\n"
        f"  SELECT\n"
        f"    chunk_id,\n"
        f"    source_uri,\n"
        f"    out.text AS text,\n"
        f"    CAST(NULL AS STRING) AS lang,\n"
        f"    STRUCT(0 AS offset_start, LENGTH(out.text) AS offset_end) AS span\n"
        f"  FROM (\n"
        f"    SELECT\n"
        f"      TO_HEX(SHA256(CONCAT(object_uri, '#img'))) AS chunk_id,\n"
        f"      object_uri AS source_uri,\n"
        f"      AI.GENERATE(\n"
        f"        STRUCT('{extract_instr}' AS instruction, OBJ.GET_ACCESS_URL(object_ref, 'r') AS ref),\n"
        f"        connection_id => '{conn_short}',\n"
        f"        output_schema => 'text STRING'\n"
        f"      ) AS out\n"
        f"    FROM {images_view}\n"
        f"  )\n"
        f"  WHERE out.text IS NOT NULL AND LENGTH(out.text) > 0\n"
        f") AS src\n"
        f"ON dst.chunk_id = src.chunk_id\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (chunk_id, source_uri, text, lang, span) VALUES (src.chunk_id, src.source_uri, src.text, src.lang, src.span);\n"
        f"INSERT INTO {cfg.table('prompt_log')} (call_id, prompt_id, prompt_version, model_id, input_hash)\n"
        f"SELECT GENERATE_UUID(), 'text_extract_v1', 'text_extract_v1', 'AI.GENERATE', TO_HEX(SHA256(o.object_uri))\n"
        f"FROM {images_view} AS o\n"
        f"LEFT JOIN {chunks} d ON TO_HEX(SHA256(CONCAT(o.object_uri, '#img'))) = d.chunk_id\n"
        f"WHERE d.chunk_id IS NULL;"
    )


# String schema for AI.GENERATE_TABLE (portable across environments)
FACTS_STRING_SCHEMA = "entity STRING, predicate STRING, value_text STRING, value_number FLOAT64, value_date STRING, confidence FLOAT64, source_span STRING"


def extract_facts_sql(cfg: Config, cst: AllConstants, prompt_id: str) -> str:
    """Idempotent MERGE that extracts typed facts via AI.GENERATE."""
    chunks = cfg.table("text_chunks")
    facts = cfg.table("facts_generic")
    conf_thresh = cst.validators.validator_conf_thresh
    prompt_log = cfg.table("prompt_log")
    conn_short = cfg.connection_id_short()
    conn_arg = f", connection_id => '{conn_short}'" if conn_short else ""
    limit_env = os.getenv("FACTS_MAX_ROWS", "").strip()
    limit_clause = (
        f" LIMIT {int(limit_env)}" if limit_env.isdigit() and int(limit_env) > 0 else ""
    )
    clamp_env = os.getenv("FACTS_MAX_CHARS", "").strip()
    clamp_chars = int(clamp_env) if clamp_env.isdigit() and int(clamp_env) > 0 else 1500
    facts_instr = _prompt_facts_instruction(cfg)
    return (
        f"MERGE {facts} AS dst\n"
        f"USING (\n"
        f"  WITH gen AS (\n"
        f"    SELECT\n"
        f"      c.source_uri AS source_uri,\n"
        f"      c.span AS source_span_struct,\n"
        f"      AI.GENERATE(\n"
        f"        STRUCT('{facts_instr}' AS instruction, SUBSTR(c.text, 1, {clamp_chars}) AS prompt),\n"
        f"        output_schema => '{FACTS_STRING_SCHEMA}'{conn_arg}\n"
        f"      ) AS out\n"
        f"    FROM {chunks} c{limit_clause}\n"
        f"  )\n"
        f"  SELECT\n"
        f"    CONCAT(source_uri, '#', CAST(source_span_struct.offset_start AS STRING), '-', CAST(source_span_struct.offset_end AS STRING)) AS fact_id,\n"
        f"    source_uri, source_span_struct,\n"
        f"    out.entity AS subject, out.predicate AS predicate, out.value_text AS object_value, out.value_number AS object_number,\n"
        f"    SAFE.PARSE_DATE('%Y-%m-%d', out.value_date) AS object_date, out.confidence AS confidence\n"
        f"  FROM gen\n"
        f"  WHERE out.confidence >= {conf_thresh}\n"
        f") AS src\n"
        f"ON dst.fact_id = src.fact_id\n"
        f"WHEN NOT MATCHED THEN\n"
        f"  INSERT (fact_id, subject, predicate, object_value, object_number, object_date, source_uri, source_span, confidence)\n"
        f"  VALUES (src.fact_id, src.subject, src.predicate, src.object_value, src.object_number, src.object_date, src.source_uri, src.source_span_struct, src.confidence);\n"
        f"INSERT INTO {prompt_log} (call_id, prompt_id, prompt_version, model_id, input_hash)\n"
        f"SELECT GENERATE_UUID(), '{prompt_id}', 'facts_v1', 'AI.GENERATE', TO_HEX(SHA256(c.text)) FROM {chunks} AS c{limit_clause};"
    )


def _vs_id_col() -> str:
    col = os.getenv("VECTOR_SEARCH_ID_COL", "id").strip().lower()
    return "row_id" if col == "row_id" else "id"


def vs_select_id(alias: str = "vs") -> str:
    """Qualified column reference to the VECTOR_SEARCH row identifier for an alias."""
    return f"{alias}.{_vs_id_col()}"


def _rid_expr_from_vs_json(vs_json_alias: str = "vs_json") -> str:
    """Extract neighbor id from VECTOR_SEARCH result JSON regardless of id variant."""
    return f"COALESCE(JSON_VALUE({vs_json_alias}, '$.id'), JSON_VALUE({vs_json_alias}, '$.row_id'))"


def _rid_join_condition(
    rid_alias: str, table_alias: str, key_col: str = "chunk_id"
) -> str:
    """Join condition that matches either raw key or its JSON string form."""
    return f"({rid_alias} = {table_alias}.{key_col} OR {rid_alias} = TO_JSON_STRING({table_alias}.{key_col}))"


@dataclass(frozen=True)
class Constraint:
    """Simple typed constraint to apply after VECTOR_SEARCH."""

    field: str
    operator: str
    value: str

    def to_sql(self) -> str:
        # Safe simple predicate. Complex predicates can be added as needed.
        if self.field.upper() == "TRUE":
            return "TRUE"
        return f"{self.field} {self.operator} {self.value}"


def vector_search_sql(
    cfg: Config,
    cst: AllConstants,
    query_embedding_expr: str,
    modality: str = "text",
    constraints: Iterable[Constraint] = (),
    k: int | None = None,
) -> str:
    """Select neighbor ids and distances with optional constraints.

    Returns columns: rid (STRING) and distance (FLOAT64).
    """
    table = cfg.table("text_emb" if modality == "text" else "image_emb")
    limit_k = k or cst.embed.vector_k
    pred = " AND ".join(c.to_sql() for c in constraints) if constraints else "TRUE"
    base_key = "chunk_id" if modality == "text" else "image_id"
    join_on = _rid_join_condition("i.rid", "e", base_key)
    return (
        f"WITH vs_raw AS (\n"
        f"  SELECT TO_JSON(vs) AS vs_json, vs.distance\n"
        f"  FROM VECTOR_SEARCH(TABLE {table}, 'vector', (SELECT {query_embedding_expr} AS vector), top_k => {limit_k}) AS vs\n"
        f"), ids AS (\n"
        f"  SELECT {_rid_expr_from_vs_json('vs_json')} AS rid, distance\n"
        f"  FROM vs_raw\n"
        f"  WHERE {_rid_expr_from_vs_json('vs_json')} IS NOT NULL\n"
        f"), joined AS (\n"
        f"  SELECT i.rid, i.distance, e.{base_key} AS id\n"
        f"  FROM ids i JOIN {table} e ON {join_on}\n"
        f")\n"
        f"SELECT rid, distance FROM joined WHERE {pred}\n"
    )


def hybrid_score_sql(
    alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.1, delta: float = 0.1
) -> str:
    """Projection for hybrid scoring with tunable weights."""
    return (
        f"(\n"
        f"  {alpha} * similarity +\n"
        f"  {beta} * IFNULL(CAST(constraints_ok AS INT64), 0) +\n"
        f"  {gamma} * 0.0 +\n"
        f"  {delta} * 0.0\n"
        f") AS hybrid_score"
    )


def precompute_neighbors_sql(
    cfg: Config, cst: AllConstants, query_id: str, embedding_expr: str
) -> str:
    """Insert top neighbors for a single query embedding into precomputed_neighbors."""
    table = cfg.table("text_emb")
    k = cst.eval.neighbor_precomp_k
    return (
        f"INSERT INTO {cfg.table('precomputed_neighbors')} (query_id, neighbor_id, neighbor_type, similarity)\n"
        f"WITH vs_raw AS (\n"
        f"  SELECT TO_JSON(vs) AS vs_json, vs.distance\n"
        f"  FROM VECTOR_SEARCH(TABLE {table}, 'vector', (SELECT {embedding_expr} AS vector), top_k => {k}) AS vs\n"
        f"), ids AS (\n"
        f"  SELECT {_rid_expr_from_vs_json('vs_json')} AS rid, distance\n"
        f"  FROM vs_raw\n"
        f"  WHERE {_rid_expr_from_vs_json('vs_json')} IS NOT NULL\n"
        f")\n"
        f"SELECT '{query_id}', rid, 'text', 1.0 - distance FROM ids\n"
    )


def precompute_neighbors_bulk_sql(cfg: Config, cst: AllConstants) -> str:
    """Iterate `hot_queries` and fill neighbors table (scripted)."""
    text_emb = cfg.table("text_emb")
    k = cst.eval.neighbor_precomp_k
    return (
        f"DECLARE v ARRAY<FLOAT64>;\n"
        f"DECLARE v_q STRING;\n"
        f"FOR rec IN (SELECT query_id, embedding FROM {cfg.table('hot_queries')}) DO\n"
        f"  SET v = rec.embedding;\n"
        f"  SET v_q = rec.query_id;\n"
        f"  INSERT INTO {cfg.table('precomputed_neighbors')} (query_id, neighbor_id, neighbor_type, similarity)\n"
        f"  WITH vs_raw AS (\n"
        f"    SELECT TO_JSON(vs) AS vs_json, vs.distance\n"
        f"    FROM VECTOR_SEARCH(TABLE {text_emb}, 'vector', (SELECT v AS vector), top_k => {k}) AS vs\n"
        f"  ), ids AS (\n"
        f"    SELECT {_rid_expr_from_vs_json('vs_json')} AS rid, distance\n"
        f"    FROM vs_raw\n"
        f"    WHERE {_rid_expr_from_vs_json('vs_json')} IS NOT NULL\n"
        f"  )\n"
        f"  SELECT v_q, rid, 'text', 1.0 - distance FROM ids;\n"
        f"END FOR;"
    )


def create_hybrid_view_sql(cfg: Config, cst: AllConstants) -> str:
    """Create or replace a view that exposes hybrid-scored edges."""
    edges = cfg.table("edges_semantic")
    return (
        f"CREATE OR REPLACE VIEW {cfg.table('v_hybrid_top')} AS\n"
        f"SELECT *, {hybrid_score_sql(cst.eval.hybrid_alpha, cst.eval.hybrid_beta, cst.eval.hybrid_gamma, cst.eval.hybrid_delta)}\n"
        f"FROM {edges}\n"
        f"ORDER BY hybrid_score DESC LIMIT 100"
    )


def retrieval_eval_sql(cfg: Config, cst: AllConstants) -> str:
    """Loop over hot_queries and write recall/precision/MRR metrics (scripted)."""
    text_emb = cfg.table("text_emb")
    k = cst.embed.vector_k
    return (
        f"DECLARE has_gt BOOL DEFAULT (SELECT COUNT(*) > 0 FROM {cfg.table('retrieval_gt')});\n"
        f"DECLARE v ARRAY<FLOAT64>;\n"
        f"DECLARE v_q STRING;\n"
        f"IF has_gt THEN\n"
        f"  FOR rec IN (SELECT query_id, embedding FROM {cfg.table('hot_queries')}) DO\n"
        f"    SET v = rec.embedding;\n"
        f"    SET v_q = rec.query_id;\n"
        f"    INSERT INTO {cfg.table('metrics_retrieval')} (query_id, k, recall_k, precision_k, mrr)\n"
        f"    WITH vs_raw AS (\n"
        f"      SELECT TO_JSON(vs) AS vs_json, vs.distance\n"
        f"      FROM VECTOR_SEARCH(TABLE {text_emb}, 'vector', (SELECT v AS vector), top_k => {k}) AS vs\n"
        f"    ), ids AS (\n"
        f"      SELECT {_rid_expr_from_vs_json('vs_json')} AS rid, distance\n"
        f"      FROM vs_raw\n"
        f"      WHERE {_rid_expr_from_vs_json('vs_json')} IS NOT NULL\n"
        f"    ), preds AS (\n"
        f"      SELECT v_q AS query_id, e.chunk_id AS item_id, ROW_NUMBER() OVER (PARTITION BY v_q ORDER BY ids.distance) AS rank\n"
        f"      FROM ids\n"
        f"      JOIN {text_emb} e ON {_rid_join_condition('ids.rid', 'e', 'chunk_id')}\n"
        f"    ), gt AS (\n"
        f"      SELECT query_id, item_id FROM {cfg.table('retrieval_gt')}\n"
        f"    ), hits AS (\n"
        f"      SELECT p.query_id, p.rank FROM preds p JOIN gt g ON g.query_id = p.query_id AND g.item_id = p.item_id\n"
        f"    ), gt_counts AS (\n"
        f"      SELECT query_id, COUNT(*) AS gt_n FROM gt GROUP BY query_id\n"
        f"    ), hit_counts AS (\n"
        f"      SELECT query_id, COUNT(*) AS hits FROM hits GROUP BY query_id\n"
        f"    ), best_hit AS (\n"
        f"      SELECT query_id, MIN(rank) AS best_rank FROM hits GROUP BY query_id\n"
        f"    ), per_query AS (\n"
        f"      SELECT '{k}' AS k, gc.query_id, IFNULL(hc.hits, 0) AS hits, gc.gt_n, IFNULL(1.0 / NULLIF(bh.best_rank, 0), 0.0) AS rr\n"
        f"      FROM gt_counts gc\n"
        f"      LEFT JOIN hit_counts hc USING (query_id)\n"
        f"      LEFT JOIN best_hit bh USING (query_id)\n"
        f"    )\n"
        f"    SELECT query_id, {k} AS k, SAFE_DIVIDE(hits, NULLIF(gt_n, 0)) AS recall_k, SAFE_DIVIDE(hits, {k}) AS precision_k, rr AS mrr FROM per_query;\n"
        f"  END FOR;\n"
        f"END IF;"
    )


def retrieval_context_sql(
    cfg: Config, cst: AllConstants, query_param: str, k: int | None = None
) -> str:
    """Return a SELECT providing (source_uri, text) context using VECTOR_SEARCH."""
    text_model = cst.embed.text_model
    k_eff = k or cst.embed.vector_k
    return (
        f"WITH q AS (\n"
        f"  SELECT ml_generate_embedding_result AS v\n"
        f"  FROM ML.GENERATE_EMBEDDING(\n"
        f"    MODEL `{text_model}`,\n"
        f"    (SELECT CAST({query_param} AS STRING) AS content),\n"
        f"    STRUCT(TRUE AS flatten_json_output)\n"
        f"  )\n"
        f"), vs_raw AS (\n"
        f"  SELECT TO_JSON(vs) AS vs_json, vs.distance\n"
        f"  FROM VECTOR_SEARCH(TABLE {cfg.table('text_emb')}, 'vector', (SELECT v FROM q), top_k => {k_eff}) AS vs\n"
        f"), ids AS (\n"
        f"  SELECT {_rid_expr_from_vs_json('vs_json')} AS rid, distance\n"
        f"  FROM vs_raw\n"
        f"  WHERE {_rid_expr_from_vs_json('vs_json')} IS NOT NULL\n"
        f")\n"
        f"SELECT c.source_uri, c.text\n"
        f"FROM ids i JOIN {cfg.table('text_emb')} e ON {_rid_join_condition('i.rid', 'e', 'chunk_id')}\n"
        f"JOIN {cfg.table('text_chunks')} c ON c.chunk_id = e.chunk_id\n"
        f"ORDER BY i.distance ASC\n"
        f"LIMIT {k_eff}"
    )


def bool_validator_sql(
    cfg: Config,
    rule_id: str,
    context_table: str,
    id_column: str,
    context_expr: str,
    *,
    sample_frac: float | None = None,
) -> str:
    """Evaluate a BOOL validator via AI.GENERATE_BOOL and log results."""
    metrics = cfg.table("metrics_validators")
    conn = cfg.connection_id_short()
    conn_arg = f", connection_id => '{conn}'" if conn else ""
    seed = AllConstants().orch.seed

    # Deterministic sampling by id via hash modulus into [0,1)
    sample_expr = f"(MOD(ABS(FARM_FINGERPRINT(CAST({id_column} AS STRING)) + {seed}), 1000000) / 1000000.0)"
    where_sample = (
        f" WHERE {sample_expr} < {sample_frac} "
        if sample_frac and sample_frac > 0.0
        else ""
    )

    # Deterministic order per id
    order_expr = f"ABS(FARM_FINGERPRINT(CONCAT(CAST({id_column} AS STRING), '{seed}')))"
    cap = AllConstants().validators.max_validators_per_item

    return (
        f"INSERT INTO {metrics} (item_id, rule_id, result_bool, score, note)\n"
        f"SELECT CAST({id_column} AS STRING), '{rule_id}',\n"
        f"       (AI.GENERATE_BOOL({context_expr}{conn_arg})).result AS result_bool,\n"
        f"       NULL AS score,\n"
        f"       NULL AS note\n"
        f"FROM (\n"
        f"  SELECT *, ROW_NUMBER() OVER (PARTITION BY {id_column} ORDER BY {order_expr}) AS _rn\n"
        f"  FROM {cfg.table(context_table)}{where_sample}\n"
        f") AS s\n"
        f"WHERE _rn <= {cap}\n"
        f"AND NOT EXISTS (SELECT 1 FROM {metrics} m WHERE m.item_id = CAST({id_column} AS STRING) AND m.rule_id = '{rule_id}');\n"
        f"INSERT INTO {cfg.table('prompt_log')} (call_id, prompt_id, prompt_version, model_id, input_hash)\n"
        f"SELECT GENERATE_UUID(), 'validator_v1', 'validator_v1', 'AI.GENERATE_BOOL', TO_HEX(SHA256(TO_JSON_STRING({context_expr})))\n"
        f"FROM (\n"
        f"  SELECT *, ROW_NUMBER() OVER (PARTITION BY {id_column} ORDER BY {order_expr}) AS _rn\n"
        f"  FROM {cfg.table(context_table)}{where_sample}\n"
        f") AS s\n"
        f"WHERE _rn <= {cap}\n"
        f"AND NOT EXISTS (SELECT 1 FROM {metrics} m WHERE m.item_id = CAST({id_column} AS STRING) AND m.rule_id = '{rule_id}');"
    )


def double_tolerance_validator_sql(
    cfg: Config,
    cst: AllConstants,
    rule_id: str,
    context_table: str,
    id_column: str,
    context_expr: str,
    ref_value_expr: str,
    tolerance_frac: float | None = None,
    *,
    sample_frac: float | None = None,
) -> str:
    """Numeric extraction check with tolerance via AI.GENERATE_DOUBLE."""
    metrics = cfg.table("metrics_validators")
    conn = cfg.connection_id_short()
    conn_arg = f", connection_id => '{conn}'" if conn else ""
    tol = tolerance_frac if tolerance_frac is not None else 0.02
    seed = AllConstants().orch.seed

    sample_expr = f"(MOD(ABS(FARM_FINGERPRINT(CAST({id_column} AS STRING)) + {seed}), 1000000) / 1000000.0)"
    where_sample = (
        f" WHERE {sample_expr} < {sample_frac} "
        if sample_frac and sample_frac > 0.0
        else ""
    )
    order_expr = f"ABS(FARM_FINGERPRINT(CONCAT(CAST({id_column} AS STRING), '{seed}')))"
    cap = AllConstants().validators.max_validators_per_item

    # AI.GENERATE_DOUBLE returns a STRUCT in some configurations; select .result for numeric
    return (
        f"INSERT INTO {metrics} (item_id, rule_id, result_bool, score, note)\n"
        f"SELECT CAST({id_column} AS STRING), '{rule_id}',\n"
        f"       ABS((AI.GENERATE_DOUBLE({context_expr}{conn_arg})).result - ({ref_value_expr})) <= ({tol}) * ({ref_value_expr}) AS result_bool,\n"
        f"       (AI.GENERATE_DOUBLE({context_expr}{conn_arg})).result AS score,\n"
        f"       'tolerance=' || CAST({tol} AS STRING) AS note\n"
        f"FROM (\n"
        f"  SELECT *, ROW_NUMBER() OVER (PARTITION BY {id_column} ORDER BY {order_expr}) AS _rn\n"
        f"  FROM {cfg.table(context_table)}{where_sample}\n"
        f") AS s\n"
        f"WHERE _rn <= {cap}\n"
        f"AND NOT EXISTS (SELECT 1 FROM {metrics} m WHERE m.item_id = CAST({id_column} AS STRING) AND m.rule_id = '{rule_id}');\n"
        f"INSERT INTO {cfg.table('prompt_log')} (call_id, prompt_id, prompt_version, model_id, input_hash)\n"
        f"SELECT GENERATE_UUID(), 'validator_v1', 'validator_v1', 'AI.GENERATE_DOUBLE', TO_HEX(SHA256(TO_JSON_STRING({context_expr})))\n"
        f"FROM (\n"
        f"  SELECT *, ROW_NUMBER() OVER (PARTITION BY {id_column} ORDER BY {order_expr}) AS _rn\n"
        f"  FROM {cfg.table(context_table)}{where_sample}\n"
        f") AS s\n"
        f"WHERE _rn <= {cap}\n"
        f"AND NOT EXISTS (SELECT 1 FROM {metrics} m WHERE m.item_id = CAST({id_column} AS STRING) AND m.rule_id = '{rule_id}');"
    )


def brand_model_match_sql(cfg: Config, rule_id: str = "rule_brand_model") -> str:
    """Check if captions mention expected brand/model tokens using AI.GENERATE_BOOL."""
    metrics = cfg.table("metrics_validators")
    conn = cfg.connection_id_short()
    conn_arg = f", connection_id => '{conn}'" if conn else ""
    seed = AllConstants().orch.seed
    cap = AllConstants().validators.max_validators_per_item
    order_expr = f"ABS(FARM_FINGERPRINT(CONCAT(CAST(image_id AS STRING), '{seed}')))"

    return (
        f"INSERT INTO {metrics} (item_id, rule_id, result_bool, score, note)\n"
        f"SELECT CAST(image_id AS STRING), '{rule_id}',\n"
        f"       (AI.GENERATE_BOOL(STRUCT(CONCAT('Does this caption mention the expected brand/model? ', caption) AS text){conn_arg})).result,\n"
        f"       NULL, NULL\n"
        f"FROM (\n"
        f"  SELECT *, ROW_NUMBER() OVER (PARTITION BY image_id ORDER BY {order_expr}) AS _rn\n"
        f"  FROM {cfg.table('image_desc')}\n"
        f") AS s WHERE _rn <= {cap}\n"
        f"AND NOT EXISTS (SELECT 1 FROM {metrics} m WHERE m.item_id = CAST(image_id AS STRING) AND m.rule_id = '{rule_id}');"
    )


def table_vs_text_total_sql(
    cfg: Config, rule_id: str = "rule_table_text_total", tolerance: float = 0.02
) -> str:
    """Compare numeric text extraction vs table value within a tolerance."""
    metrics = cfg.table("metrics_validators")
    conn = cfg.connection_id_short()
    conn_arg = f", connection_id => '{conn}'" if conn else ""
    seed = AllConstants().orch.seed
    cap = AllConstants().validators.max_validators_per_item
    order_expr = f"ABS(FARM_FINGERPRINT(CONCAT(CAST(fact_id AS STRING), '{seed}')))"

    return (
        f"INSERT INTO {metrics} (item_id, rule_id, result_bool, score, note)\n"
        f"SELECT CAST(fact_id AS STRING), '{rule_id}',\n"
        f"       ABS((AI.GENERATE_DOUBLE(STRUCT(object_value AS text){conn_arg})).result - object_number) <= ({tolerance}) * object_number,\n"
        f"       (AI.GENERATE_DOUBLE(STRUCT(object_value AS text){conn_arg})).result,\n"
        f"       'tolerance=' || CAST({tolerance} AS STRING)\n"
        f"FROM (\n"
        f"  SELECT *, ROW_NUMBER() OVER (PARTITION BY fact_id ORDER BY {order_expr}) AS _rn\n"
        f"  FROM {cfg.table('facts_generic')}\n"
        f") AS s WHERE _rn <= {cap}\n"
        f"AND NOT EXISTS (SELECT 1 FROM {metrics} m WHERE m.item_id = CAST(fact_id AS STRING) AND m.rule_id = '{rule_id}');"
    )


def forecast_sql(cfg: Config, cst: AllConstants, kpi_id_param: str) -> str:
    """Run AI.FORECAST on a KPI series with backtests and insert results."""
    series = cfg.table("kpi_series")
    forecast = cfg.table("kpi_forecast")
    horizon = cst.forecast.forecast_horizon_days
    backtests = cst.forecast.forecast_backtest_splits
    return (
        f"DECLARE pts INT64 DEFAULT (SELECT COUNTIF(ts IS NOT NULL) FROM {series} WHERE kpi_id = {kpi_id_param} AND ts IS NOT NULL);\n"
        f"IF pts >= 3 THEN\n"
        f"INSERT INTO {forecast} (run_id, kpi_id, ts, predicted, lower, upper, backtest)\n"
        f"SELECT GENERATE_UUID(), {kpi_id_param}, DATE(forecast_timestamp) AS ts, forecast_value AS predicted,\n"
        f"       prediction_interval_lower_bound AS lower, prediction_interval_upper_bound AS upper, FALSE AS backtest\n"
        f"FROM AI.FORECAST(\n"
        f"  (SELECT ts, value FROM {series} WHERE kpi_id = {kpi_id_param} AND ts IS NOT NULL),\n"
        f"  data_col => 'value', timestamp_col => 'ts', horizon => {horizon}\n"
        f")\n"
        f"WHERE forecast_timestamp IS NOT NULL;\n"
        f"FOR rec IN (\n"
        f"  SELECT ts FROM {series} WHERE kpi_id = {kpi_id_param} AND ts IS NOT NULL ORDER BY ts DESC LIMIT {backtests}\n"
        f") DO\n"
        f"  IF rec.ts IS NOT NULL THEN\n"
        f"  INSERT INTO {forecast} (run_id, kpi_id, ts, predicted, lower, upper, backtest)\n"
        f"  SELECT GENERATE_UUID(), {kpi_id_param}, DATE(forecast_timestamp) AS ts, forecast_value AS predicted,\n"
        f"         prediction_interval_lower_bound AS lower, prediction_interval_upper_bound AS upper, TRUE AS backtest\n"
        f"  FROM AI.FORECAST(\n"
        f"    (SELECT ts, value FROM {series} WHERE kpi_id = {kpi_id_param} AND ts IS NOT NULL AND ts <= rec.ts),\n"
        f"    data_col => 'value', timestamp_col => 'ts', horizon => {horizon}\n"
        f"  )\n"
        f"  WHERE forecast_timestamp IS NOT NULL;\n"
        f"  END IF;\n"
        f"END FOR;\n"
        f"END IF;"
    )


def schemas_ddl_sql(cfg: Config) -> str:
    """Return a BigQuery SQL script that applies canonical DDL (CREATE IF NOT EXISTS).

    Based on SCHEMAS.md. Adds optional Object Tables + union views.
    """
    ds = cfg.dataset_id
    p = cfg.project_id

    # Optional Object Tables DDL (create-if-missing) + union views to normalize access
    object_tables_sql = ""
    if cfg.object_tables_enabled:
        conn_full = cfg.connection_id_full()
        img = (cfg.gcs_images_prefix or "").strip()
        pdf = (cfg.gcs_pdfs_prefix or "").strip()
        scr = (cfg.gcs_screens_prefix or "").strip()
        parts: list[str] = []

        # Images
        if img:
            parts.append(
                f"CREATE EXTERNAL TABLE IF NOT EXISTS {cfg.table('objects_images_ot')}\n"
                f"WITH CONNECTION `{conn_full}`\n"
                f"OPTIONS (\n"
                f"  object_metadata = 'SIMPLE',\n"
                f"  uris = ['{img}*']\n"
                f");\n"
            )
            parts.append(
                f"CREATE OR REPLACE VIEW {cfg.table('v_objects_images')} AS\n"
                f"WITH all_uris AS (\n"
                f"  SELECT uri AS object_uri FROM {cfg.table('objects_images_ot')}\n"
                f"  UNION DISTINCT\n"
                f"  SELECT object_uri FROM {cfg.table('objects_images')}\n"
                f")\n"
                f"SELECT object_uri, OBJ.MAKE_REF(object_uri, '{conn_full}') AS object_ref FROM all_uris;\n"
            )
        else:
            parts.append(
                f"CREATE OR REPLACE VIEW {cfg.table('v_objects_images')} AS\n"
                f"SELECT object_uri, OBJ.MAKE_REF(object_uri, '{conn_full}') AS object_ref FROM {cfg.table('objects_images')};\n"
            )

        # PDFs
        if pdf:
            parts.append(
                f"CREATE EXTERNAL TABLE IF NOT EXISTS {cfg.table('objects_pdfs_ot')}\n"
                f"WITH CONNECTION `{conn_full}`\n"
                f"OPTIONS (\n"
                f"  object_metadata = 'SIMPLE',\n"
                f"  uris = ['{pdf}*']\n"
                f");\n"
            )
            parts.append(
                f"CREATE OR REPLACE VIEW {cfg.table('v_objects_pdfs')} AS\n"
                f"WITH all_uris AS (\n"
                f"  SELECT uri AS object_uri FROM {cfg.table('objects_pdfs_ot')}\n"
                f"  UNION DISTINCT\n"
                f"  SELECT object_uri FROM {cfg.table('objects_pdfs')}\n"
                f")\n"
                f"SELECT object_uri, OBJ.MAKE_REF(object_uri, '{conn_full}') AS object_ref FROM all_uris;\n"
            )
        else:
            parts.append(
                f"CREATE OR REPLACE VIEW {cfg.table('v_objects_pdfs')} AS\n"
                f"SELECT object_uri, OBJ.MAKE_REF(object_uri, '{conn_full}') AS object_ref FROM {cfg.table('objects_pdfs')};\n"
            )

        # Screens
        if scr:
            parts.append(
                f"CREATE EXTERNAL TABLE IF NOT EXISTS {cfg.table('objects_screens_ot')}\n"
                f"WITH CONNECTION `{conn_full}`\n"
                f"OPTIONS (\n"
                f"  object_metadata = 'SIMPLE',\n"
                f"  uris = ['{scr}*']\n"
                f");\n"
            )
            parts.append(
                f"CREATE OR REPLACE VIEW {cfg.table('v_objects_screens')} AS\n"
                f"WITH all_uris AS (\n"
                f"  SELECT uri AS object_uri FROM {cfg.table('objects_screens_ot')}\n"
                f"  UNION DISTINCT\n"
                f"  SELECT object_uri FROM {cfg.table('objects_screens')}\n"
                f")\n"
                f"SELECT object_uri, OBJ.MAKE_REF(object_uri, '{conn_full}') AS object_ref FROM all_uris;\n"
            )
        else:
            parts.append(
                f"CREATE OR REPLACE VIEW {cfg.table('v_objects_screens')} AS\n"
                f"SELECT object_uri, OBJ.MAKE_REF(object_uri, '{conn_full}') AS object_ref FROM {cfg.table('objects_screens')};\n"
            )

        object_tables_sql = "".join(parts)
    else:
        # Views backed only by legacy tables; still expose object_ref built from uri
        conn_full = cfg.connection_id_full()
        object_tables_sql = (
            f"CREATE OR REPLACE VIEW {cfg.table('v_objects_images')} AS\n"
            f"SELECT object_uri, OBJ.MAKE_REF(object_uri, '{conn_full}') AS object_ref FROM {cfg.table('objects_images')};\n"
            f"CREATE OR REPLACE VIEW {cfg.table('v_objects_pdfs')} AS\n"
            f"SELECT object_uri, OBJ.MAKE_REF(object_uri, '{conn_full}') AS object_ref FROM {cfg.table('objects_pdfs')};\n"
            f"CREATE OR REPLACE VIEW {cfg.table('v_objects_screens')} AS\n"
            f"SELECT object_uri, OBJ.MAKE_REF(object_uri, '{conn_full}') AS object_ref FROM {cfg.table('objects_screens')};\n"
        )

    return (
        # Remote models for embeddings (create-or-replace)
        f"CREATE OR REPLACE MODEL {cfg.table('text_embedding_model')}\n"
        f"  REMOTE WITH CONNECTION `{cfg.connection_id_full()}`\n"
        f"  OPTIONS (ENDPOINT = 'text-embedding-004');\n"
        f"CREATE OR REPLACE MODEL {cfg.table('image_embedding_model')}\n"
        f"  REMOTE WITH CONNECTION `{cfg.connection_id_full()}`\n"
        f"  OPTIONS (ENDPOINT = 'multimodalembedding@001');\n"
        # Legacy object uri tables (metadata placeholders)
        f"CREATE TABLE IF NOT EXISTS {cfg.table('objects_images')} (\n"
        f"  object_uri STRING NOT NULL,\n"
        f"  media_type STRING NOT NULL,\n"
        f"  object_ref JSON,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f");\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('objects_pdfs')} (\n"
        f"  object_uri STRING NOT NULL,\n"
        f"  media_type STRING NOT NULL,\n"
        f"  object_ref JSON,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f");\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('objects_screens')} (\n"
        f"  object_uri STRING NOT NULL,\n"
        f"  media_type STRING NOT NULL,\n"
        f"  object_ref JSON,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f");\n" + object_tables_sql +
        # Text chunks and image descriptors
        f"CREATE TABLE IF NOT EXISTS {cfg.table('text_chunks')} (\n"
        f"  chunk_id STRING NOT NULL,\n"
        f"  source_uri STRING NOT NULL,\n"
        f"  text STRING NOT NULL,\n"
        f"  lang STRING,\n"
        f"  span STRUCT<offset_start INT64, offset_end INT64>,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (chunk_id) NOT ENFORCED\n"
        f");\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('image_desc')} (\n"
        f"  image_id STRING NOT NULL,\n"
        f"  source_uri STRING NOT NULL,\n"
        f"  caption STRING,\n"
        f"  dominant_colors ARRAY<STRING>,\n"
        f"  width_px INT64,\n"
        f"  height_px INT64,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (image_id) NOT ENFORCED\n"
        f");\n"
        # Embeddings
        f"CREATE TABLE IF NOT EXISTS {cfg.table('text_emb')} (\n"
        f"  chunk_id STRING NOT NULL,\n"
        f"  model_id STRING NOT NULL,\n"
        f"  vector ARRAY<FLOAT64>,\n"
        f"  vector_dim INT64 NOT NULL,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (chunk_id) NOT ENFORCED\n"
        f") PARTITION BY DATE(created_ts) CLUSTER BY model_id;\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('image_emb')} (\n"
        f"  image_id STRING NOT NULL,\n"
        f"  model_id STRING NOT NULL,\n"
        f"  vector ARRAY<FLOAT64>,\n"
        f"  vector_dim INT64 NOT NULL,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (image_id) NOT ENFORCED\n"
        f") PARTITION BY DATE(created_ts) CLUSTER BY model_id;\n"
        # Facts
        f"CREATE TABLE IF NOT EXISTS {cfg.table('facts_generic')} (\n"
        f"  fact_id STRING NOT NULL,\n"
        f"  subject STRING NOT NULL,\n"
        f"  predicate STRING NOT NULL,\n"
        f"  object_value STRING,\n"
        f"  object_number FLOAT64,\n"
        f"  object_date DATE,\n"
        f"  source_uri STRING NOT NULL,\n"
        f"  source_span STRUCT<offset_start INT64, offset_end INT64>,\n"
        f"  confidence FLOAT64,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (fact_id) NOT ENFORCED\n"
        f");\n"
        # Edges
        f"CREATE TABLE IF NOT EXISTS {cfg.table('edges_semantic')} (\n"
        f"  edge_id STRING NOT NULL,\n"
        f"  left_id STRING NOT NULL,\n"
        f"  right_id STRING NOT NULL,\n"
        f"  left_type STRING NOT NULL,\n"
        f"  right_type STRING NOT NULL,\n"
        f"  similarity FLOAT64 NOT NULL,\n"
        f"  constraints_ok BOOL,\n"
        f"  constraint_notes STRING,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (edge_id) NOT ENFORCED\n"
        f") PARTITION BY DATE(created_ts) CLUSTER BY left_id, right_id;\n"
        # Answers
        f"CREATE TABLE IF NOT EXISTS {cfg.table('answers')} (\n"
        f"  answer_id STRING NOT NULL,\n"
        f"  query STRING NOT NULL,\n"
        f"  summary STRING NOT NULL,\n"
        f"  citations ARRAY<STRING>,\n"
        f"  validators ARRAY<STRUCT<name STRING, result BOOL, note STRING>>,\n"
        f"  confidence FLOAT64,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (answer_id) NOT ENFORCED\n"
        f");\n"
        # KPI series and forecast
        f"CREATE TABLE IF NOT EXISTS {cfg.table('kpi_series')} (\n"
        f"  kpi_id STRING NOT NULL,\n"
        f"  ts DATE NOT NULL,\n"
        f"  value FLOAT64 NOT NULL,\n"
        f"  tags ARRAY<STRING>,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (kpi_id, ts) NOT ENFORCED\n"
        f");\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('kpi_forecast')} (\n"
        f"  run_id STRING NOT NULL,\n"
        f"  kpi_id STRING NOT NULL,\n"
        f"  ts DATE NOT NULL,\n"
        f"  predicted FLOAT64 NOT NULL,\n"
        f"  lower FLOAT64,\n"
        f"  upper FLOAT64,\n"
        f"  backtest BOOL,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (run_id, kpi_id, ts) NOT ENFORCED\n"
        f");\n"
        # Views (examples + diagnostics)
        f"CREATE OR REPLACE VIEW {cfg.table('v_answer_cards')} AS\n"
        f"SELECT a.answer_id, a.query, a.summary,\n"
        f"       ARRAY_LENGTH(a.citations) AS num_citations,\n"
        f"       a.confidence, a.created_ts\n"
        f"FROM {cfg.table('answers')} a;\n"
        f"CREATE OR REPLACE VIEW {cfg.table('v_index_health')} AS\n"
        f"SELECT 'text_emb' AS table_name,\n"
        f"       (SELECT COUNT(*) FROM {cfg.table('text_emb')}) AS row_count,\n"
        f"       (SELECT COUNT(*) FROM {cfg.table('INFORMATION_SCHEMA.VECTOR_INDEXES')} WHERE index_name = 'text_emb_idx') > 0 AS has_index\n"
        f"UNION ALL\n"
        f"SELECT 'image_emb' AS table_name,\n"
        f"       (SELECT COUNT(*) FROM {cfg.table('image_emb')}) AS row_count,\n"
        f"       (SELECT COUNT(*) FROM {cfg.table('INFORMATION_SCHEMA.VECTOR_INDEXES')} WHERE index_name = 'image_emb_idx') > 0 AS has_index;\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('metrics_validators')} (\n"
        f"  item_id STRING NOT NULL,\n"
        f"  rule_id STRING NOT NULL,\n"
        f"  result_bool BOOL,\n"
        f"  score FLOAT64,\n"
        f"  note STRING,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f") PARTITION BY DATE(created_ts);\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('metrics_retrieval')} (\n"
        f"  query_id STRING NOT NULL,\n"
        f"  k INT64 NOT NULL,\n"
        f"  recall_k FLOAT64,\n"
        f"  precision_k FLOAT64,\n"
        f"  mrr FLOAT64,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f") PARTITION BY DATE(created_ts);\n"
        f"CREATE OR REPLACE VIEW {cfg.table('v_validator_coverage')} AS\n"
        f"SELECT rule_id, COUNT(*) AS n, AVG(CAST(result_bool AS INT64)) AS pass_rate\n"
        f"FROM {cfg.table('metrics_validators')}\n"
        f"GROUP BY rule_id;\n"
        f"CREATE OR REPLACE VIEW {cfg.table('v_retrieval_metrics')} AS\n"
        f"SELECT k, AVG(recall_k) AS recall_k, AVG(precision_k) AS precision_k, AVG(mrr) AS mrr\n"
        f"FROM {cfg.table('metrics_retrieval')}\n"
        f"GROUP BY k;\n"
        # Metrics & logs
        f"CREATE TABLE IF NOT EXISTS {cfg.table('prompt_log')} (\n"
        f"  call_id STRING NOT NULL,\n"
        f"  prompt_id STRING NOT NULL,\n"
        f"  prompt_version STRING NOT NULL,\n"
        f"  model_id STRING NOT NULL,\n"
        f"  input_hash STRING,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (call_id) NOT ENFORCED\n"
        f");\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('metrics_validators')} (\n"
        f"  item_id STRING NOT NULL,\n"
        f"  rule_id STRING NOT NULL,\n"
        f"  result_bool BOOL,\n"
        f"  score FLOAT64,\n"
        f"  note STRING,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f") PARTITION BY DATE(created_ts);\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('metrics_retrieval')} (\n"
        f"  query_id STRING NOT NULL,\n"
        f"  k INT64 NOT NULL,\n"
        f"  recall_k FLOAT64,\n"
        f"  precision_k FLOAT64,\n"
        f"  mrr FLOAT64,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f") PARTITION BY DATE(created_ts);\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('metrics_cost')} (\n"
        f"  run_id STRING NOT NULL,\n"
        f"  statement_label STRING NOT NULL,\n"
        f"  bytes_processed INT64,\n"
        f"  elapsed_ms FLOAT64,\n"
        f"  dry_run BOOL,\n"
        f"  statement_type STRING,\n"
        f"  cache_hit BOOL,\n"
        f"  ok BOOL,\n"
        f"  error STRING,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f") PARTITION BY DATE(created_ts);\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('metrics_index_state')} (\n"
        f"  index_name STRING NOT NULL,\n"
        f"  last_row_count INT64 NOT NULL,\n"
        f"  updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (index_name) NOT ENFORCED\n"
        f");\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('hot_queries')} (\n"
        f"  query_id STRING NOT NULL,\n"
        f"  query_text STRING,\n"
        f"  embedding ARRAY<FLOAT64>,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),\n"
        f"  PRIMARY KEY (query_id) NOT ENFORCED\n"
        f");\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('precomputed_neighbors')} (\n"
        f"  query_id STRING NOT NULL,\n"
        f"  neighbor_id STRING NOT NULL,\n"
        f"  neighbor_type STRING NOT NULL,\n"
        f"  similarity FLOAT64,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f");\n"
        f"CREATE TABLE IF NOT EXISTS {cfg.table('retrieval_gt')} (\n"
        f"  query_id STRING NOT NULL,\n"
        f"  item_id STRING NOT NULL,\n"
        f"  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()\n"
        f");\n"
    )
