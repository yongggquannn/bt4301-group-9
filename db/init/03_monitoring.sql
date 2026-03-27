CREATE TABLE IF NOT EXISTS processed.model_monitoring_results (
    result_id               BIGSERIAL     PRIMARY KEY,  
    -- monitoring run timestamp
    monitored_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    -- baseline/current comparison windows (time-window reference for monitoring run)
    baseline_window_start   TIMESTAMPTZ   NOT NULL,
    baseline_window_end     TIMESTAMPTZ   NOT NULL,
    current_window_start    TIMESTAMPTZ   NOT NULL,
    current_window_end      TIMESTAMPTZ   NOT NULL,
    -- model performance monitoring (degradation); nullable when AUC cannot be computed
    baseline_auc            NUMERIC(8,6),
    current_auc             NUMERIC(8,6),
    auc_delta               NUMERIC(8,6),
    -- drift threshold summary
    max_psi                 NUMERIC(10,6) NOT NULL,
    -- alert outcome and reasons
    breached                BOOLEAN       NOT NULL DEFAULT FALSE,
    breached_reasons        TEXT,
    -- source of baseline AUC used for degradation comparison
    baseline_auc_source     TEXT          NOT NULL DEFAULT 'historical_baseline_window',
    -- cohort construction and sample size for auditability
    cohort_strategy         TEXT          NOT NULL DEFAULT 'time_window',
    baseline_row_count      INT           NOT NULL DEFAULT 0,
    current_row_count       INT           NOT NULL DEFAULT 0,
    -- feature-level drift details (JSON: {feature_name: psi_value})
    psi_by_feature          JSONB         NOT NULL
);
