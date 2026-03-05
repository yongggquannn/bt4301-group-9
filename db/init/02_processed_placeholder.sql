-- Placeholder for feature table / processed layer
CREATE TABLE IF NOT EXISTS processed.feature_table (
  feature_id     BIGSERIAL PRIMARY KEY,
  source_id      BIGINT,
  features       JSONB,
  created_at     TIMESTAMPTZ DEFAULT NOW()
);