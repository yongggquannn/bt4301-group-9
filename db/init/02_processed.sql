CREATE TABLE IF NOT EXISTS processed.customer_features (
    msno                          TEXT          NOT NULL,
    is_churn                      SMALLINT      NOT NULL,
    -- demographics (raw.members)
    city                          SMALLINT,
    bd                            SMALLINT,
    gender                        VARCHAR(10),
    registered_via                SMALLINT,
    registration_init_time        INT,
    -- transaction aggregates (raw.transactions)
    transaction_count             INT,
    renewal_count                 INT,
    cancel_count                  INT,
    total_amount_paid             NUMERIC,
    avg_plan_days                 NUMERIC(8,2),
    latest_payment_method_id      SMALLINT,
    latest_is_auto_renew          SMALLINT,
    latest_membership_expire_date INT,
    -- user log aggregates (raw.user_logs)
    num_active_days               INT,
    total_secs                    NUMERIC,
    avg_total_secs                NUMERIC(12,4),
    total_num_songs               INT,
    avg_num_songs                 NUMERIC(10,4),
    total_num_unq                 INT,
    avg_num_unq                   NUMERIC(10,4),
    feature_created_at            TIMESTAMPTZ   DEFAULT NOW(),
    PRIMARY KEY (msno)
);

CREATE TABLE IF NOT EXISTS processed.data_lineage (
    feature_name TEXT NOT NULL,
    source_table TEXT NOT NULL,
    transformation_rule TEXT NOT NULL,
    snapshot_id UUID,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS processed.data_watermarks (
    watermark_id  SERIAL      PRIMARY KEY,
    table_name    TEXT        NOT NULL,
    row_count     INTEGER    NOT NULL,
    content_hash  TEXT        NOT NULL,
    pipeline_run_id TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

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

CREATE TABLE IF NOT EXISTS processed.churn_predictions (
    customer_id         TEXT        NOT NULL,
    churn_probability   NUMERIC     NOT NULL,
    risk_tier           VARCHAR(16) NOT NULL,
    scored_at           TIMESTAMPTZ NOT NULL,
    shap_values         JSONB,
    feature_snapshot_id UUID,
    PRIMARY KEY (customer_id, scored_at)
);

-- Workstream E — stronger validation and contracts.
-- One row per rule per validate_data run; blocking failures fail the task.
CREATE TABLE IF NOT EXISTS processed.validation_results (
    result_id   BIGSERIAL   PRIMARY KEY,
    run_id      TEXT        NOT NULL,
    rule_name   TEXT        NOT NULL,
    severity    VARCHAR(16) NOT NULL,
    status      VARCHAR(16) NOT NULL,
    detail      JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_validation_results_run_id
    ON processed.validation_results (run_id);

CREATE INDEX IF NOT EXISTS idx_validation_results_created_at
    ON processed.validation_results (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_validation_results_rule_name
    ON processed.validation_results (rule_name, created_at DESC);
