CREATE TABLE IF NOT EXISTS raw.members (
  msno                    TEXT          NOT NULL,
  city                    SMALLINT,
  bd                      SMALLINT,
  gender                  VARCHAR(10),
  registered_via          SMALLINT,
  registration_init_time  INT,
  ingestion_timestamp     TIMESTAMPTZ   DEFAULT NOW(),
  PRIMARY KEY (msno)
);

CREATE TABLE IF NOT EXISTS raw.transactions (
  msno                    TEXT          NOT NULL,
  payment_method_id       SMALLINT,
  payment_plan_days       INT,
  plan_list_price         INT,
  actual_amount_paid      INT,
  is_auto_renew           SMALLINT,
  transaction_date        INT,
  membership_expire_date  INT,
  is_cancel               SMALLINT,
  ingestion_timestamp     TIMESTAMPTZ   DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.user_logs (
  msno                TEXT          NOT NULL,
  date                INT,
  num_25              INT,
  num_50              INT,
  num_75              INT,
  num_985             INT,
  num_100             INT,
  num_unq             INT,
  total_secs          NUMERIC,
  ingestion_timestamp TIMESTAMPTZ   DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.train (
  msno                TEXT      NOT NULL,
  is_churn            SMALLINT  NOT NULL,
  ingestion_timestamp TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (msno)
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions.churn_predictions (
  customer_id       TEXT          NOT NULL,
  churn_probability NUMERIC(5,4),
  risk_tier         VARCHAR(10),
  scored_at         TIMESTAMPTZ   DEFAULT NOW(),
  PRIMARY KEY (customer_id)
);

