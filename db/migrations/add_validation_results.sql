-- Workstream E — DataOps stronger validation & contracts.
-- Creates processed.validation_results so the validate_data step can persist
-- rule-level outcomes (schema/freshness/coverage/distribution + legacy smoke
-- checks) with severity and JSONB detail for baselines and drift info.

CREATE TABLE IF NOT EXISTS processed.validation_results (
    result_id   BIGSERIAL   PRIMARY KEY,
    run_id      TEXT        NOT NULL,
    rule_name   TEXT        NOT NULL,
    severity    VARCHAR(16) NOT NULL,   -- 'warning' | 'blocking'
    status      VARCHAR(16) NOT NULL,   -- 'pass' | 'fail' | 'skip'
    detail      JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_validation_results_run_id
    ON processed.validation_results (run_id);

CREATE INDEX IF NOT EXISTS idx_validation_results_created_at
    ON processed.validation_results (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_validation_results_rule_name
    ON processed.validation_results (rule_name, created_at DESC);
