-- file: src/sql/templates.sql
-- Consolidated SQL templates for MSKG
-- Sections: answers, embedding, facts, forecast, index, validators

-- Answers section
INSERT INTO `${DATASET_ID}.answers` (answer_id, query, summary, citations, validators, confidence)
WITH ctx AS (
  SELECT source_uri, text FROM `${DATASET_ID}.text_chunks`
  LIMIT ${TOPK_CONTEXT}
), agg AS (
  SELECT
    @query_text AS query,
    ARRAY_AGG(SUBSTR(REGEXP_REPLACE(REGEXP_REPLACE(text, r'\n|\r', ' '), '"', ' '), 1, 96)) AS snippets,
    ARRAY_AGG(DISTINCT ctx.source_uri) AS allowed_uris
  FROM ctx
), gen AS (
  SELECT AI.GENERATE(
    STRUCT(a.query AS query, a.snippets AS context_snippets, a.allowed_uris AS allowed_uris),
    output_schema => 'summary STRING, citations ARRAY<STRING>, confidence FLOAT64, validators ARRAY<STRUCT<name STRING, result BOOL, note STRING>>',
    connection_id => '${PROJECT_ID}.${REGION}.gemini',
    model_params => JSON '{"generation_config":{"max_output_tokens":24,"temperature":0}}'
  ) AS out, a.allowed_uris AS allowed_uris
  FROM agg a
), v AS (
  SELECT ARRAY_AGG(STRUCT(name AS name, result AS result, note AS note)) AS validators
  FROM (
    SELECT rule_id AS name, LOGICAL_AND(result_bool) AS result, ANY_VALUE(note) AS note
    FROM `${DATASET_ID}.metrics_validators`
    GROUP BY rule_id
  )
)
SELECT
  GENERATE_UUID() AS answer_id,
  @query_text AS query,
  COALESCE(out.summary, '') AS summary,
  (SELECT ARRAY( SELECT uri FROM UNNEST(out.citations) AS uri WHERE uri IN UNNEST(allowed_uris) )) AS citations,
  COALESCE((SELECT validators FROM v), (SELECT ARRAY(SELECT AS STRUCT name, result, note FROM UNNEST(out.validators)))) AS validators,
  GREATEST(0.0, LEAST(1.0, out.confidence)) AS confidence
FROM gen;

INSERT INTO `${DATASET_ID}.prompt_log` (call_id, prompt_id, prompt_version, model_id, input_hash)
SELECT GENERATE_UUID(), 'answer_v1', 'answer_v1', 'AI.GENERATE', TO_HEX(SHA256(CAST(@query_text AS STRING)));

-- Text embeddings
MERGE `${DATASET_ID}.text_emb` AS dst
USING (
  SELECT
    chunk_id, '${TEXT_EMBED_MODEL}' AS model_id,
    ml_generate_embedding_result AS vector,
    ${TEXT_VECTOR_DIM} AS vector_dim
  FROM ML.GENERATE_EMBEDDING(
    MODEL `${TEXT_EMBED_MODEL}`,
    (SELECT chunk_id, text AS content FROM `${DATASET_ID}.text_chunks`),
    STRUCT(TRUE AS flatten_json_output)
  )
  WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0
) AS src
ON dst.chunk_id = src.chunk_id
WHEN NOT MATCHED THEN
  INSERT (chunk_id, model_id, vector, vector_dim) VALUES (src.chunk_id, src.model_id, src.vector, src.vector_dim);

-- Image embeddings
MERGE `${DATASET_ID}.image_emb` AS dst
USING (
  SELECT
    image_id, '${IMAGE_EMBED_MODEL}' AS model_id,
    ml_generate_embedding_result AS vector,
    ${IMAGE_VECTOR_DIM} AS vector_dim
  FROM ML.GENERATE_EMBEDDING(
    MODEL `${IMAGE_EMBED_MODEL}`,
    (SELECT image_id, caption AS content FROM `${DATASET_ID}.image_desc` WHERE caption IS NOT NULL),
    STRUCT(TRUE AS flatten_json_output)
  )
  WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0
) AS src
ON dst.image_id = src.image_id
WHEN NOT MATCHED THEN
  INSERT (image_id, model_id, vector, vector_dim) VALUES (src.image_id, src.model_id, src.vector, src.vector_dim);

-- Extract facts from text chunks
MERGE `${DATASET_ID}.facts_generic` AS dst
USING (
  WITH gen AS (
    SELECT
      c.source_uri AS source_uri,
      c.span AS source_span_struct,
      AI.GENERATE(
        STRUCT('Extract facts: entity, predicate, values (text/number/date) and confidence from the snippet' AS instruction, c.text AS prompt),
        connection_id => '${PROJECT_ID}.${REGION}.gemini',
        output_schema => 'entity STRING, predicate STRING, value_text STRING, value_number FLOAT64, value_date STRING, confidence FLOAT64, source_span STRING'
      ) AS out
    FROM `${DATASET_ID}.text_chunks` c
  )
  SELECT
    CONCAT(source_uri, '#', CAST(source_span_struct.offset_start AS STRING), '-', CAST(source_span_struct.offset_end AS STRING)) AS fact_id,
    source_uri,
    source_span_struct,
    out.entity AS subject,
    out.predicate,
    out.value_text AS object_value,
    out.value_number AS object_number,
    SAFE.PARSE_DATE('%Y-%m-%d', out.value_date) AS object_date,
    out.confidence
  FROM gen
  WHERE out.confidence >= ${VALIDATOR_CONF_THRESH}
) AS src
ON dst.fact_id = src.fact_id
WHEN NOT MATCHED THEN
  INSERT (fact_id, subject, predicate, object_value, object_number, object_date, source_uri, source_span, confidence)
  VALUES (src.fact_id, src.subject, src.predicate, src.object_value, src.object_number, src.object_date, src.source_uri, src.source_span_struct, src.confidence);

INSERT INTO `${DATASET_ID}.prompt_log` (call_id, prompt_id, prompt_version, model_id, input_hash)
SELECT GENERATE_UUID(), 'facts_v1', 'facts_v1', '${PROJECT_ID}.${REGION}.gemini', TO_HEX(SHA256(c.text))
FROM `${DATASET_ID}.text_chunks` AS c;

-- KPI forecasting with backtesting
DECLARE _split_ts DATE;

INSERT INTO `${DATASET_ID}.kpi_forecast` (run_id, kpi_id, ts, predicted, lower, upper, backtest)
SELECT GENERATE_UUID(), @kpi, DATE(forecast_timestamp) AS ts, forecast_value AS predicted,
       prediction_interval_lower_bound AS lower, prediction_interval_upper_bound AS upper, FALSE AS backtest
FROM AI.FORECAST(
  (SELECT ts, value FROM `${DATASET_ID}.kpi_series` WHERE kpi_id = @kpi),
  data_col => 'value', timestamp_col => 'ts', horizon => ${FORECAST_HORIZON_DAYS}
);

FOR _split_ts IN (
  SELECT ts FROM `${DATASET_ID}.kpi_series` WHERE kpi_id = @kpi ORDER BY ts DESC LIMIT ${FORECAST_BACKTEST_SPLITS}
) DO
  INSERT INTO `${DATASET_ID}.kpi_forecast` (run_id, kpi_id, ts, predicted, lower, upper, backtest)
  SELECT GENERATE_UUID(), @kpi, DATE(forecast_timestamp) AS ts, forecast_value AS predicted,
         prediction_interval_lower_bound AS lower, prediction_interval_upper_bound AS upper, TRUE AS backtest
  FROM AI.FORECAST(
    (SELECT ts, value FROM `${DATASET_ID}.kpi_series` WHERE kpi_id = @kpi AND ts <= _split_ts),
    data_col => 'value', timestamp_col => 'ts', horizon => ${FORECAST_HORIZON_DAYS}
  );
END FOR;

CREATE OR REPLACE VIEW `${DATASET_ID}.v_kpi_cards` AS
SELECT kpi_id, ts, predicted, lower, upper
FROM `${DATASET_ID}.kpi_forecast`
WHERE backtest = FALSE
ORDER BY ts DESC
LIMIT 100;

-- Vector index management
DECLARE has_text_vec BOOL DEFAULT (SELECT COUNTIF(ARRAY_LENGTH(vector) > 0) > 0 FROM `${DATASET_ID}.text_emb`);
DECLARE has_image_vec BOOL DEFAULT (SELECT COUNTIF(ARRAY_LENGTH(vector) > 0) > 0 FROM `${DATASET_ID}.image_emb`);
DECLARE has_text_idx BOOL DEFAULT (SELECT COUNT(*) > 0 FROM `${DATASET_ID}.INFORMATION_SCHEMA.VECTOR_INDEXES` WHERE index_name = 'text_emb_idx');
DECLARE has_image_idx BOOL DEFAULT (SELECT COUNT(*) > 0 FROM `${DATASET_ID}.INFORMATION_SCHEMA.VECTOR_INDEXES` WHERE index_name = 'image_emb_idx');
DECLARE text_rows INT64 DEFAULT (SELECT COUNT(*) FROM `${DATASET_ID}.text_emb`);
DECLARE image_rows INT64 DEFAULT (SELECT COUNT(*) FROM `${DATASET_ID}.image_emb`);
DECLARE prev_text_rows INT64 DEFAULT COALESCE((SELECT last_row_count FROM `${DATASET_ID}.metrics_index_state` WHERE index_name = 'text_emb_idx'), 0);
DECLARE prev_image_rows INT64 DEFAULT COALESCE((SELECT last_row_count FROM `${DATASET_ID}.metrics_index_state` WHERE index_name = 'image_emb_idx'), 0);
DECLARE grow_text_pct FLOAT64 DEFAULT SAFE_DIVIDE(100.0 * (text_rows - prev_text_rows), GREATEST(prev_text_rows, 1));
DECLARE grow_image_pct FLOAT64 DEFAULT SAFE_DIVIDE(100.0 * (image_rows - prev_image_rows), GREATEST(prev_image_rows, 1));
DECLARE min_rows INT64 DEFAULT 1;

-- Clean up empty vectors
DELETE FROM `${DATASET_ID}.text_emb` WHERE vector IS NULL OR ARRAY_LENGTH(vector) = 0;
DELETE FROM `${DATASET_ID}.image_emb` WHERE vector IS NULL OR ARRAY_LENGTH(vector) = 0;

-- Create initial indexes if needed
IF has_text_vec AND NOT has_text_idx AND text_rows >= min_rows THEN
  CREATE VECTOR INDEX `${DATASET_ID}.text_emb_idx`
  ON `${DATASET_ID}.text_emb` (vector)
  OPTIONS( index_type = 'IVF', distance_type = 'COSINE' );
END IF;

IF has_image_vec AND NOT has_image_idx AND image_rows >= min_rows THEN
  CREATE VECTOR INDEX `${DATASET_ID}.image_emb_idx`
  ON `${DATASET_ID}.image_emb` (vector)
  OPTIONS( index_type = 'IVF', distance_type = 'COSINE' );
END IF;

-- Rebuild indexes if growth exceeds threshold
IF has_text_vec AND has_text_idx AND text_rows >= min_rows AND grow_text_pct > 15 THEN
  DROP INDEX `${DATASET_ID}.text_emb_idx`;
  CREATE VECTOR INDEX `${DATASET_ID}.text_emb_idx` ON `${DATASET_ID}.text_emb` (vector) OPTIONS( index_type = 'IVF', distance_type = 'COSINE' );
END IF;
IF has_image_vec AND has_image_idx AND image_rows >= min_rows AND grow_image_pct > 15 THEN
  DROP INDEX `${DATASET_ID}.image_emb_idx`;
  CREATE VECTOR INDEX `${DATASET_ID}.image_emb_idx` ON `${DATASET_ID}.image_emb` (vector) OPTIONS( index_type = 'IVF', distance_type = 'COSINE' );
END IF;

-- Update index state tracking
MERGE `${DATASET_ID}.metrics_index_state` dst USING (SELECT 'text_emb_idx' AS index_name, (SELECT COUNT(*) FROM `${DATASET_ID}.text_emb`) AS last_row_count) src ON dst.index_name = src.index_name
WHEN MATCHED THEN UPDATE SET last_row_count = src.last_row_count, updated_ts = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (index_name, last_row_count) VALUES (src.index_name, src.last_row_count);
MERGE `${DATASET_ID}.metrics_index_state` dst USING (SELECT 'image_emb_idx' AS index_name, (SELECT COUNT(*) FROM `${DATASET_ID}.image_emb`) AS last_row_count) src ON dst.index_name = src.index_name
WHEN MATCHED THEN UPDATE SET last_row_count = src.last_row_count, updated_ts = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (index_name, last_row_count) VALUES (src.index_name, src.last_row_count);

-- Data quality validators
DECLARE _seed INT64 DEFAULT 42;

-- Color matching validation for images
INSERT INTO `${DATASET_ID}.metrics_validators` (item_id, rule_id, result_bool, score, note)
SELECT CAST(image_id AS STRING), 'rule_color_match',
       (AI.GENERATE_BOOL(STRUCT(caption AS text), connection_id => '${PROJECT_ID}.${REGION}.gemini')).result AS result_bool,
       NULL AS score,
       NULL AS note
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY image_id ORDER BY RAND(_seed)) AS _rn
  FROM `${DATASET_ID}.image_desc` WHERE RAND(ABS(FARM_FINGERPRINT(CAST(image_id AS STRING)) + _seed)) < ${MAX_VALIDATION_SAMPLE_FRAC}
) AS s
WHERE _rn <= ${MAX_VALIDATORS_PER_ITEM}
AND NOT EXISTS (
  SELECT 1 FROM `${DATASET_ID}.metrics_validators` m
  WHERE m.item_id = CAST(s.image_id AS STRING) AND m.rule_id = 'rule_color_match'
);

INSERT INTO `${DATASET_ID}.prompt_log` (call_id, prompt_id, prompt_version, model_id, input_hash)
SELECT GENERATE_UUID(), 'validator_v1', 'validator_v1', 'AI.GENERATE_BOOL', TO_HEX(SHA256(TO_JSON_STRING(STRUCT(caption AS text))))
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY image_id ORDER BY RAND(_seed)) AS _rn
  FROM `${DATASET_ID}.image_desc` WHERE RAND(ABS(FARM_FINGERPRINT(CAST(image_id AS STRING)) + _seed)) < ${MAX_VALIDATION_SAMPLE_FRAC}
) AS s
WHERE _rn <= ${MAX_VALIDATORS_PER_ITEM}
AND NOT EXISTS (
  SELECT 1 FROM `${DATASET_ID}.metrics_validators` m
  WHERE m.item_id = CAST(s.image_id AS STRING) AND m.rule_id = 'rule_color_match'
);

-- Numerical accuracy validation for facts
INSERT INTO `${DATASET_ID}.metrics_validators` (item_id, rule_id, result_bool, score, note)
SELECT CAST(fact_id AS STRING), 'rule_total',
       ABS((AI.GENERATE_DOUBLE(STRUCT(object_value AS text), connection_id => '${PROJECT_ID}.${REGION}.gemini')).result - (object_number)) <= (0.02) * (object_number) AS result_bool,
       (AI.GENERATE_DOUBLE(STRUCT(object_value AS text), connection_id => '${PROJECT_ID}.${REGION}.gemini')).result AS score,
       'tolerance=0.02' AS note
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY fact_id ORDER BY RAND(_seed)) AS _rn
  FROM `${DATASET_ID}.facts_generic` WHERE RAND(ABS(FARM_FINGERPRINT(CAST(fact_id AS STRING)) + _seed)) < ${MAX_VALIDATION_SAMPLE_FRAC}
) AS s
WHERE _rn <= ${MAX_VALIDATORS_PER_ITEM}
AND NOT EXISTS (
  SELECT 1 FROM `${DATASET_ID}.metrics_validators` m
  WHERE m.item_id = CAST(s.fact_id AS STRING) AND m.rule_id = 'rule_total'
);

INSERT INTO `${DATASET_ID}.prompt_log` (call_id, prompt_id, prompt_version, model_id, input_hash)
SELECT GENERATE_UUID(), 'validator_v1', 'validator_v1', 'AI.GENERATE_DOUBLE', TO_HEX(SHA256(TO_JSON_STRING(STRUCT(object_value AS text))))
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY fact_id ORDER BY RAND(_seed)) AS _rn
  FROM `${DATASET_ID}.facts_generic` WHERE RAND(ABS(FARM_FINGERPRINT(CAST(fact_id AS STRING)) + _seed)) < ${MAX_VALIDATION_SAMPLE_FRAC}
) AS s
WHERE _rn <= ${MAX_VALIDATORS_PER_ITEM}
AND NOT EXISTS (
  SELECT 1 FROM `${DATASET_ID}.metrics_validators` m
  WHERE m.item_id = CAST(s.fact_id AS STRING) AND m.rule_id = 'rule_total'
); 