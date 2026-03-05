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
