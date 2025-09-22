-- Create vector indexes for text and image embeddings
-- Only creates indexes if vectors exist and rebuilds when data grows significantly

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

-- Clean up any invalid or empty vectors before indexing
DELETE FROM `${DATASET_ID}.text_emb` WHERE vector IS NULL OR ARRAY_LENGTH(vector) = 0;
DELETE FROM `${DATASET_ID}.image_emb` WHERE vector IS NULL OR ARRAY_LENGTH(vector) = 0;

-- Create text embedding index if we have vectors but no index yet
IF has_text_vec AND NOT has_text_idx AND text_rows >= min_rows THEN
  CREATE VECTOR INDEX `${DATASET_ID}.text_emb_idx`
  ON `${DATASET_ID}.text_emb` (vector)
  OPTIONS( index_type = 'IVF', distance_type = 'COSINE' );
END IF;

-- Create image embedding index if we have vectors but no index yet
IF has_image_vec AND NOT has_image_idx AND image_rows >= min_rows THEN
  CREATE VECTOR INDEX `${DATASET_ID}.image_emb_idx`
  ON `${DATASET_ID}.image_emb` (vector)
  OPTIONS( index_type = 'IVF', distance_type = 'COSINE' );
END IF;

-- Rebuild text index if data has grown by more than 15%
IF has_text_vec AND has_text_idx AND text_rows >= min_rows AND grow_text_pct > 15 THEN
  DROP INDEX `${DATASET_ID}.text_emb_idx`;
  CREATE VECTOR INDEX `${DATASET_ID}.text_emb_idx` ON `${DATASET_ID}.text_emb` (vector) OPTIONS( index_type = 'IVF', distance_type = 'COSINE' );
END IF;

-- Rebuild image index if data has grown by more than 15%
IF has_image_vec AND has_image_idx AND image_rows >= min_rows AND grow_image_pct > 15 THEN
  DROP INDEX `${DATASET_ID}.image_emb_idx`;
  CREATE VECTOR INDEX `${DATASET_ID}.image_emb_idx` ON `${DATASET_ID}.image_emb` (vector) OPTIONS( index_type = 'IVF', distance_type = 'COSINE' );
END IF;

-- Update tracking metrics for text index
MERGE `${DATASET_ID}.metrics_index_state` dst USING (SELECT 'text_emb_idx' AS index_name, (SELECT COUNT(*) FROM `${DATASET_ID}.text_emb`) AS last_row_count) src ON dst.index_name = src.index_name
WHEN MATCHED THEN UPDATE SET last_row_count = src.last_row_count, updated_ts = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (index_name, last_row_count) VALUES (src.index_name, src.last_row_count);

-- Update tracking metrics for image index
MERGE `${DATASET_ID}.metrics_index_state` dst USING (SELECT 'image_emb_idx' AS index_name, (SELECT COUNT(*) FROM `${DATASET_ID}.image_emb`) AS last_row_count) src ON dst.index_name = src.index_name
WHEN MATCHED THEN UPDATE SET last_row_count = src.last_row_count, updated_ts = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (index_name, last_row_count) VALUES (src.index_name, src.last_row_count);