-- Migration: add feature_snapshots table and versioning columns.
-- Safe to run multiple times (IF NOT EXISTS / IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS processed.feature_snapshots (
    snapshot_id         UUID        PRIMARY KEY,
    pipeline_run_id     TEXT,
    build_started_at    TIMESTAMPTZ NOT NULL,
    build_completed_at  TIMESTAMPTZ,
    source_watermark_id INTEGER     REFERENCES processed.data_watermarks(watermark_id),
    row_count           INTEGER,
    content_hash        TEXT,
    feature_count       INTEGER     NOT NULL,
    feature_names       JSONB       NOT NULL,
    status              VARCHAR(16) NOT NULL DEFAULT 'building',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE processed.data_lineage
    ADD COLUMN IF NOT EXISTS snapshot_id UUID;

ALTER TABLE processed.churn_predictions
    ADD COLUMN IF NOT EXISTS feature_snapshot_id UUID;
