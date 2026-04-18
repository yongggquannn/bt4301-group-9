-- Workstream F — MLOps monitoring & promotion quality.
--
-- Adds two groups of columns:
-- 1. Provenance on processed.churn_predictions (model_version, threshold_version)
--    so every scored row can be traced back to the exact model registry version
--    and classification-threshold policy in force.
-- 2. Richer monitoring fields on processed.model_monitoring_results: calibration
--    error / bins, score distribution, segment-level PSI, labelled-sample count,
--    and an explicit status ('ok' | 'breached' | 'insufficient_data').

-- --- Provenance on churn_predictions -------------------------------------
ALTER TABLE processed.churn_predictions
    ADD COLUMN IF NOT EXISTS model_version     TEXT,
    ADD COLUMN IF NOT EXISTS threshold_version TEXT;

-- --- Monitoring extras on model_monitoring_results -----------------------
ALTER TABLE processed.model_monitoring_results
    ADD COLUMN IF NOT EXISTS status                 TEXT    NOT NULL DEFAULT 'ok',
    ADD COLUMN IF NOT EXISTS brier_score            NUMERIC(10,6),
    ADD COLUMN IF NOT EXISTS max_calibration_error  NUMERIC(10,6),
    ADD COLUMN IF NOT EXISTS calibration_bins       JSONB,
    ADD COLUMN IF NOT EXISTS score_distribution     JSONB,
    ADD COLUMN IF NOT EXISTS segment_psi            JSONB,
    ADD COLUMN IF NOT EXISTS labeled_sample_count   INT     NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS min_sample_threshold   INT     NOT NULL DEFAULT 0;

-- Friendly index so retraining DAG can quickly locate the latest status row.
CREATE INDEX IF NOT EXISTS idx_model_monitoring_results_monitored_at
    ON processed.model_monitoring_results (monitored_at DESC);
