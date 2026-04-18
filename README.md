# BT4301 Group Project

BT4301 Business Analytics Solutions Development and Deployment — Group 9 DataOps/MLOps project.

## Repository structure

Matches the Canvas submission layout (Week 13 zip):

```
├── docs/                    # Submission: docs subfolder
│   ├── project_report.docx       # Group project report (Word DOCX)
│   └── peer_learning_slides.pptx # Slide deck for peer learning session
│
├── source/                  # Submission: source subfolder
│   ├── common/                  # Shared utilities (dag_utils, db, monitoring_utils)
│   ├── dataops/                 # DataOps pipeline scripts and Airflow DAGs
│   ├── mlops/                   # MLOps pipeline scripts
│   ├── webapp/                  # FastAPI churn-risk web application
│   └── tests/                   # Integration and unit tests
│
├── data/                    # Submission: data subfolder
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed/cleaned datasets
│
├── README.md
└── requirements.txt
```

**Note:** Remove `.gitkeep` files from `docs/`, `source/dataops/`, `source/mlops/`, `data/raw/`, and `data/processed/` before creating the final submission zip.

## Planning

For parallel execution planning across app, DataOps, and MLOps enhancements, see:

- `docs/parallel_enhancement_plan.md`

Use this plan to split ownership, sequence higher-impact upgrades first, and track cross-workstream dependencies.

---

## Shared Modules (`source/common/`)

These utilities are imported by all Airflow DAGs and pipeline scripts — do not run them directly.

| Module                | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `dag_utils.py`        | `run_python_script()` — subprocess runner used by all DAGs     |
| `db.py`               | `get_connection()`, `get_db_config()` — PostgreSQL helpers     |
| `monitoring_utils.py` | `compute_psi()`, `roc_auc_binary()` — drift computation        |

---

## Data Setup

The raw datasets (~1.9 GB) come from the [KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) on Kaggle and are too large to commit to Git.

### Prerequisites

1. Install the Kaggle CLI:

```
 pip install kaggle
```

2. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token** and save the downloaded `kaggle.json`:

- **macOS/Linux:** save to `~/.kaggle/kaggle.json`, then run `chmod 600 ~/.kaggle/kaggle.json`
- **Windows:** save to `C:\Users\<YourUsername>\.kaggle\kaggle.json`

3. Accept the [competition rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules) (required for the download to work).

### Download

**macOS/Linux:**

```bash
bash data/download_data.sh
```

**Windows (PowerShell — requires Git Bash or WSL):**

```powershell
bash data/download_data.sh
```

> If you don't have Git Bash or WSL, download the four CSV files manually from Kaggle and place them in `data/raw/`.

This downloads and unzips the following files into `data/raw/`:

| File                  | Description                                 |
| --------------------- | ------------------------------------------- |
| `train_v2.csv`        | Training labels — whether each user churned |
| `members_v3.csv`      | User demographic and registration info      |
| `transactions_v2.csv` | Payment transaction history                 |
| `user_logs_v2.csv`    | Daily listening activity logs               |

---

## Quick Start (Docker Compose — Full Stack)

Spin up the entire stack (PostgreSQL, Airflow, MLflow, web app) with a single command.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- KKBox data downloaded to `data/raw/` (see Data Setup above)
- ~4 GB free RAM (Airflow + ML dependencies)

### Steps

```bash
# 1. Create your .env (only needed once)
cp .env.example .env

# 2. Build and start all services
docker compose up --build
```

First build takes several minutes (installing ML dependencies in the Airflow image). Subsequent starts are fast.

### Access Points

| Service    | URL                                            | Credentials         |
| ---------- | ---------------------------------------------- | ------------------- |
| Airflow UI | [http://localhost:8080](http://localhost:8080) | admin / admin       |
| MLflow UI  | [http://localhost:5001](http://localhost:5001) | —                   |
| Web App    | [http://localhost:8000](http://localhost:8000) | —                   |
| PostgreSQL | `localhost:5432`                               | bt4301 / bt4301pass |

### Running the Pipelines

1. Open the Airflow UI at [http://localhost:8080](http://localhost:8080)
2. Trigger `dataops_e2e_pipeline` to run the full DataOps chain (ingest → EDA)
3. After DataOps completes, trigger `daily_churn_scoring` to generate predictions
4. Visit [http://localhost:8000/dashboard](http://localhost:8000/dashboard) to see scored customers

### Stopping

```bash
docker compose down       # stop containers (keeps data)
docker compose down -v    # stop and remove volumes (clean slate)
```

---

## Running the Pipeline (Manual Setup)

### Step 1 — Start PostgreSQL & MLflow

```bash
docker compose up -d
```

This starts a PostgreSQL 15 container (`bt4301_postgres`) and the MLflow tracking server (`bt4301_mlflow`) at [http://localhost:5001](http://localhost:5001). PostgreSQL schemas and tables are automatically created from `db/init/`. Wait until both containers are healthy:

```bash
docker ps
```

Look for `(healthy)` in the STATUS column. This usually takes about 15 seconds.

> **Already ran this before?** If you previously started the container and see a "table not found" error later, the init scripts may not have re-run (Docker only runs them once on first volume creation). Fix it manually:
>
> ```bash
> docker exec -i bt4301_postgres psql -U bt4301 -d kkbox < db/init/00_schemas.sql
> docker exec -i bt4301_postgres psql -U bt4301 -d kkbox < db/init/01_raw_tables.sql
> docker exec -i bt4301_postgres psql -U bt4301 -d kkbox < db/init/02_processed.sql
> docker exec -i bt4301_postgres psql -U bt4301 -d kkbox < db/init/03_mlflow_db.sql
> docker exec -i bt4301_postgres psql -U bt4301 -d kkbox < db/init/03_monitoring.sql
> ```

---

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Set database environment variables

These tell the Python scripts how to connect to the database. Set them **once per terminal session** before running any scripts.

**macOS/Linux (bash/zsh):**

```bash
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=5432
export POSTGRES_DB=kkbox
export POSTGRES_USER=bt4301
export POSTGRES_PASSWORD=bt4301pass
```

**Windows — PowerShell:**

```powershell
$env:POSTGRES_HOST="127.0.0.1"
$env:POSTGRES_PORT="5432"
$env:POSTGRES_DB="kkbox"
$env:POSTGRES_USER="bt4301"
$env:POSTGRES_PASSWORD="bt4301pass"
```

**Windows — Command Prompt (CMD):**

```cmd
set POSTGRES_HOST=127.0.0.1
set POSTGRES_PORT=5432
set POSTGRES_DB=kkbox
set POSTGRES_USER=bt4301
set POSTGRES_PASSWORD=bt4301pass
```

---

### Step 4 — Run the DataOps pipeline

Run these scripts **in order** from the project root. The commands are the same on macOS and Windows.

```bash
python source/dataops/load_raw_data.py
python source/dataops/cleanse_data.py
python source/dataops/build_customer_features.py
python source/dataops/generate_lineage.py
python source/dataops/run_eda.py
python source/dataops/generate_eda_images_report.py
```

What each script does:

| Script                          | What it does                                                            |
| ------------------------------- | ----------------------------------------------------------------------- |
| `load_raw_data.py`              | Loads `data/raw/*.csv` into the `raw.*` database tables                 |
| `cleanse_data.py`               | Cleans and prepares data; writes `data/processed/df_train_final.csv`    |
| `build_customer_features.py`    | Refreshes `processed.customer_features` (one row per customer, 21 cols) |
| `generate_lineage.py`           | Refreshes `processed.data_lineage` (22 feature-level lineage records)   |
| `run_eda.py`                    | Runs exploratory analysis; writes outputs to `data/processed/eda/`      |
| `generate_eda_images_report.py` | Generates 13 EDA charts + HTML report in `data/processed/eda/`          |

**Feature versioning (Workstream D):** `build_customer_features.py` now creates a UUID snapshot in `processed.feature_snapshots` for each feature build, linking to the source watermark and recording content hash + row count. Downstream scripts (`generate_lineage.py`, `train_model.py`, `score_churn.py`) attach this snapshot ID to their outputs so every lineage row, MLflow training run, and churn prediction can be traced back to the exact feature build that produced them.

**If upgrading an existing database**, run the migration first:

```bash
docker exec -i bt4301_postgres psql -U bt4301 -d kkbox < db/migrations/add_feature_snapshots.sql
```

---

### Step 5 — Feature selection and importance (MLflow)

Runs permutation importance analysis, selects a final non-redundant feature set, and logs the results to MLflow.

```bash
python source/mlops/feature_selection.py
```

Outputs are written to `docs/artifacts/` and logged to the MLflow tracking server at `http://localhost:5001`.

---

### Step 6 — Handle class imbalance (US-18)

Runs two MLflow experiments comparing SMOTE oversampling vs. `class_weight="balanced"` model training.

```bash
python source/mlops/train_class_imbalance.py
```

Evidence artifacts are written to `docs/artifacts/` and logged to MLflow (including `chosen_strategy.json` for the next step).

---

### Step 7 — Train and compare models (US-10)

Trains Logistic Regression, XGBoost, and MLP on `processed.customer_features` using the feature list from Step 5 and (if present) the imbalance strategy from Step 6. Each MLflow run now logs the full fitted serving pipeline (`pre` + `clf`) together with an MLflow input signature so the exact same artifact can be used later for production scoring. Comparison artifacts are written to `docs/artifacts/` (e.g. `model_comparison.csv`, `best_model.json`).

**Prerequisites:** Steps 4–6 recommended (Step 6 can be skipped—training falls back to a default imbalance strategy if `chosen_strategy.json` is missing).

```bash
python source/mlops/train_model.py
```

Quick smoke test (one model, subsampled rows):

```bash
python source/mlops/train_model.py --models logistic_regression --sample-rows 5000
```

Verification:

- confirm the winning run in MLflow contains a `model` artifact with a signature
- inspect `docs/artifacts/best_model.json` for `model_artifact_type: "serving_pipeline"` and `input_features`

---

### Step 8 — Hyperparameter tuning (US-17)

Runs an Optuna study (30 trials by default) to find optimal XGBoost hyperparameters. Each trial is logged as a nested MLflow run.

**Prerequisites:** Steps 4–7 (feature store, feature selection, imbalance strategy, and baseline model).

```bash
python source/mlops/tune_hyperparams.py
```

Quick test with fewer trials:

```bash
python source/mlops/tune_hyperparams.py --n-trials 5 --sample-rows 5000
```

Outputs in `docs/artifacts/`:

- `best_hyperparams.json` — best params and AUC-ROC
- `optimization_curve.png` — Optuna optimisation history
- `param_importance.png` — hyperparameter importance
- `improvement_summary.md` — baseline vs tuned comparison

---

### Step 9 — Misclassification analysis (US-22)

Analyzes validation-set errors to show confusion matrix breakdown (FN vs FP), compares feature distributions between misclassified and correctly classified records, and generates an insights report.

**Prerequisites:** Steps 4–7 are recommended, especially `docs/artifacts/final_feature_set.json` from Step 5.

```bash
python source/mlops/analyze_misclassifications.py
```

Optional smoke test on fewer rows:

```bash
python source/mlops/analyze_misclassifications.py --sample-rows 5000 --model xgboost
```

Outputs:

- `docs/misclassification_analysis.md`
- `docs/artifacts/confusion_breakdown.json`
- `docs/artifacts/misclassified_cases.csv`
- `docs/artifacts/numeric_distribution_comparison.csv`
- `docs/artifacts/categorical_distribution_comparison.csv`
- `docs/artifacts/feature_distribution_plot.png`

---

### Step 10 — Register best model in MLflow Model Registry (US-12 / US-20)

Registers the best model (from US-10 training) into the MLflow Model Registry as `KKBox-Churn-Classifier` and applies a champion-challenger promotion rule. Registration now fails fast if the winning run does not contain a valid MLflow input signature, which protects the downstream scoring pipeline from train/serve drift.

Promotion rule:

- if there is no current Production champion, the new version becomes the initial champion
- if `new_model_auc > champion_auc + threshold`, the challenger is promoted to Production
- otherwise the current champion stays in Production and the new version remains visible in Staging as a challenger

**Prerequisites:** Docker containers from Step 1 must be running (`docker compose up -d`).

```bash
python source/mlops/register_model.py
```

Optional: require a minimum AUC improvement before promotion.

```bash
python source/mlops/register_model.py --promotion-threshold 0.005
```

Evidence is saved to:

- `docs/artifacts/model_registry.json`
- `docs/artifacts/champion_challenger_registry.json`
- `docs/champion_challenger_evidence.md`

Visit the MLflow UI at [http://localhost:5001/#/models/KKBox-Churn-Classifier](http://localhost:5001/#/models/KKBox-Churn-Classifier) to view:

- the current Production champion
- the new challenger version
- model version tags and description recording the promotion decision
- version history across earlier and current sprints

---

### Step 11 — Generate daily predictions from the production model (US-13)

Runs the scoring pipeline against the production model. The scorer resolves `MLFLOW_MODEL_URI` if provided, otherwise it uses the MLflow registry model `models:/KKBox-Churn-Classifier/Production`. It loads the registered sklearn pipeline directly, enforces the MLflow signature column order, and fails fast if the production model is unavailable or the feature table is missing required columns. There is no local fallback model during inference.

```bash
python source/mlops/score_churn.py all
```

This writes predictions into `processed.churn_predictions`, which are then used by the monitoring DAG. The intermediate model handoff is saved to `data/scoring/production_model.json`.

Verification:

```bash
python source/mlops/train_model.py --models logistic_regression --sample-rows 5000
python source/mlops/register_model.py
python source/mlops/score_churn.py all
```

Then confirm:

- `data/scoring/production_model.json` exists and lists the resolved model URI plus ordered input columns
- `processed.churn_predictions` contains fresh rows after scoring

---

### Step 12 — Churn risk web app (US-16 / US-23)

A FastAPI web app with three pages: a customer lookup, a customer retention detail page with action recommendations and business-friendly explanations, and a top-50 retention queue dashboard.

**Prerequisites:** Steps 1–11 (PostgreSQL running, `processed.churn_predictions` populated). Step 13 (SHAP) is optional — the app falls back to global feature importance if per-customer SHAP values are not yet stored.

**Install additional dependencies:**

```bash
pip install fastapi "uvicorn[standard]" jinja2
```

**Start the web app:**

```bash
python -m uvicorn source.webapp.app:app --host 0.0.0.0 --port 8000 --reload
```

**Pages:**

| URL                                   | Description                                                    |
| ------------------------------------- | -------------------------------------------------------------- |
| `http://localhost:8000/`              | Retention desk homepage — search a customer and open the dashboard |
| `http://localhost:8000/customer/<id>` | Churn probability (%), risk tier badge, recommended action, customer segment, and top 3 SHAP features |
| `http://localhost:8000/dashboard`     | Manager view — portfolio summary cards, churn distribution chart, segment breakdowns (tenure, payment, cancellation, usage), model monitoring status, and top 50 retention queue |

**Verify:**

1. Open [http://localhost:8000](http://localhost:8000) — the retention desk homepage renders with the customer search bar and dashboard link.
2. Enter a valid customer ID and click "Open Customer" → redirects to `/customer/<id>` showing probability, risk badge, intervention priority, recommended action, and explanation bullets.
3. Visit [http://localhost:8000/dashboard](http://localhost:8000/dashboard) → portfolio cards (high/medium/low/total) render at the top, followed by model freshness and monitoring status badges, the churn probability distribution chart, segment breakdowns (Customer Tenure, Payment Behavior, Cancellation History, Usage Intensity), and the top 50 retention queue.
4. Use the risk-tier and customer-segment filters, then click column headers to confirm the table still sorts correctly.
5. Click a customer ID link in the dashboard → navigates to the matching detail page with recommendation content.
6. Enter a non-existent customer ID → friendly "Customer not found." error message.

**JSON API (backward compatible):**

```bash
curl http://localhost:8000/customer/<customer_id>/churn-risk
```

Returns `churn_probability` (float), `risk_tier` (High/Medium/Low), `top_3_features`, and `top_3_shap` (per-customer SHAP values if available).

The API also returns:

- `recommended_action`
- `intervention_priority`
- `customer_segment`
- `business_explanations`

---

### Step 13 — SHAP explainability (US-19)

Generates SHAP values for the production churn model, producing a summary plot (global feature importance) and waterfall plots for three sample customers (high/medium/low risk). SHAP values are also stored per-prediction in the database and logged to MLflow.

**Prerequisites:** Steps 1–11 (PostgreSQL running, `processed.churn_predictions` populated, model trained and registered).

**If upgrading an existing database** (table already exists without the `shap_values` column), run the migration first:

```bash
docker exec -i bt4301_postgres psql -U bt4301 -d kkbox < db/migrations/add_shap_values.sql
```

**Run the SHAP script:**

```bash
python source/mlops/explain_shap.py
```

**CLI options:**

| Flag               | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `--tracking-uri`   | MLflow tracking server URI (default: `http://localhost:5001`)    |
| `--top-k N`        | Number of top SHAP features to store per prediction (default: 5) |
| `--skip-db-update` | Skip writing SHAP values back to the database                    |

**Artifacts produced (in `docs/artifacts/`):**

- `shap_summary.png` — SHAP beeswarm summary plot
- `shap_waterfall_high.png` — Waterfall plot for a high-risk customer
- `shap_waterfall_medium.png` — Waterfall plot for a medium-risk customer
- `shap_waterfall_low.png` — Waterfall plot for a low-risk customer

**Verify DB update:**

```sql
SELECT customer_id, shap_values
FROM processed.churn_predictions
WHERE shap_values IS NOT NULL
LIMIT 3;
```

---

### Step 14 — Model governance artifacts

Governance artifacts are written automatically during `train_model.py` (Step 7) via `source/mlops/governance.py` — no separate script is needed.

**What `governance.py` generates:**

| Artifact                 | Description                                         |
| ------------------------ | --------------------------------------------------- |
| `model_card.md`          | Training metadata, feature list, validation metrics |
| `feature_list_types.csv` | Feature names and data types                        |
| `fairness_gender.csv`    | Churn rates and prediction rates by gender          |
| `fairness_age_band.csv`  | Churn rates and prediction rates by age band        |

All artifacts are written to `docs/artifacts/` and logged to MLflow under the best model's run. Evidence guide: `docs/governance_evidence.md`

---

### Step 15 — Weekly monitoring checks (US-14)

Run `US-15` first so the best model artifacts are ready, then continue with registration, scoring, and weekly monitoring.

The weekly monitoring DAG checks feature drift and model degradation.

It:

- computes PSI for the top 5 monitorable features
- compares current AUC against a baseline AUC
- logs results into `processed.model_monitoring_results`
- triggers an alert task if `PSI > 0.2` or `AUC delta > 0.05`

After triggering `daily_churn_scoring` and `weekly_model_monitoring` in Airflow, connect to PostgreSQL and run:

```sql
SELECT
  monitored_at,
  baseline_auc_source,
  cohort_strategy,
  baseline_auc,
  current_auc,
  auc_delta,
  max_psi,
  breached,
  breached_reasons,
  baseline_row_count,
  current_row_count,
  psi_by_feature
FROM processed.model_monitoring_results
ORDER BY monitored_at DESC
LIMIT 10;
```

This query is the main evidence query for `US-14` because it shows the stored PSI values, AUC comparison, alert decision, and row counts for each monitoring run.

Monitoring evidence guide:

- `docs/monitoring_evidence.md`

---

### Step 16 — Automated retraining DAG (US-21)

The automated retraining DAG reads the latest monitoring result and only runs the full retraining path when drift or degradation exceeds the US-14 thresholds.

Expected Airflow task chain:

- `check_drift_results -> retrain_if_needed -> evaluate -> register`

Conditional skip path:

- `check_drift_results -> skip_retraining`

Retraining trigger rule:

- retrain only if `max_psi > 0.2`
- or if `auc_delta > 0.05`

Evidence artifacts:

- `docs/artifacts/retraining_evaluation.json`
- `docs/artifacts/retraining_decision.json`
- `docs/retraining_evidence.md`

The DAG uses `source/mlops/register_model.py`, so the final promotion step still goes through the champion-challenger gate from US-20.

---

### Step 17 — Validate database outputs

Connect to the database and run the validation queries.

**macOS/Linux:**

```bash
docker exec -it bt4301_postgres psql -U bt4301 -d kkbox
```

**Windows (PowerShell or CMD):**

```powershell
docker exec -it bt4301_postgres psql -U bt4301 -d kkbox
```

Then paste and run:

```sql
SELECT COUNT(*) AS raw_train_rows FROM raw.train;
SELECT COUNT(*) AS customer_feature_rows FROM processed.customer_features;
SELECT COUNT(*) AS lineage_rows FROM processed.data_lineage;

WITH feature_cols AS (
  SELECT column_name
  FROM information_schema.columns
  WHERE table_schema = 'processed'
    AND table_name = 'customer_features'
    AND column_name <> 'msno'
),
lineage_feats AS (
  SELECT DISTINCT feature_name FROM processed.data_lineage
)
SELECT
  (SELECT COUNT(*) FROM feature_cols fc
   LEFT JOIN lineage_feats lf ON fc.column_name = lf.feature_name
   WHERE lf.feature_name IS NULL) AS missing_in_lineage,
  (SELECT COUNT(*) FROM lineage_feats lf
   LEFT JOIN feature_cols fc ON lf.feature_name = fc.column_name
   WHERE fc.column_name IS NULL) AS extra_in_lineage;
```

Expected results:

- `raw_train_rows` equals `customer_feature_rows`
- `missing_in_lineage = 0`
- `extra_in_lineage = 0`

**Feature versioning linkage (Workstream D):** verify that snapshots, lineage, and predictions are linked:

```sql
-- Latest feature snapshot
SELECT snapshot_id, status, row_count, content_hash, source_watermark_id
FROM processed.feature_snapshots
ORDER BY build_completed_at DESC LIMIT 1;

-- Lineage rows linked to latest snapshot
SELECT COUNT(*) AS linked_lineage_rows
FROM processed.data_lineage
WHERE snapshot_id IS NOT NULL;

-- Predictions linked to a feature snapshot
SELECT COUNT(*) AS linked_prediction_rows
FROM processed.churn_predictions
WHERE feature_snapshot_id IS NOT NULL;
```

---

### Step 18 — Data validation with severity + persistence (Workstream E)

`source/dataops/validate_data.py` runs declarative rules from
`docs/cleansing_rules.yaml` (schema, freshness, coverage, distribution,
plus legacy smoke checks) and writes every outcome to
`processed.validation_results`. Each rule is tagged `warning` or
`blocking`; a blocking failure fails the Airflow task.

**One-time migration** (only needed on pre-existing databases — a fresh
`docker compose up` already picks up the table from `db/init/02_processed.sql`):

```bash
docker exec -i bt4301_postgres psql -U bt4301 -d kkbox \
  < db/migrations/add_validation_results.sql
```

**Run:**

```bash
python source/dataops/validate_data.py
# or, point at a different config file:
python source/dataops/validate_data.py --config docs/cleansing_rules.yaml
```

On the happy path every rule prints `[PASS]` and the script exits 0.
If a blocking rule fails, the script exits 1 and the Airflow task fails.

**Inspect results:**

```sql
-- Latest run outcomes
SELECT rule_name, severity, status, created_at
FROM processed.validation_results
WHERE run_id = (
  SELECT run_id FROM processed.validation_results
  ORDER BY created_at DESC LIMIT 1
)
ORDER BY severity DESC, rule_name;

-- Rules that have ever failed blocking
SELECT rule_name, MAX(created_at) AS last_failed_at
FROM processed.validation_results
WHERE severity = 'blocking' AND status = 'fail'
GROUP BY rule_name
ORDER BY last_failed_at DESC;
```

**Run the unit tests:**

```bash
python -m pytest source/tests/test_validation_rules.py -v
```

Tests use a fake cursor — no Postgres connection is required.

---

### Step 19 — Monitoring & promotion quality (Workstream F)

Workstream F hardens the monitoring and champion-challenger flows:

- **Provenance on every prediction** — `processed.churn_predictions` now records `model_version` (from the MLflow registry) and `threshold_version` (the classification policy, currently `v1` at 0.5) alongside `feature_snapshot_id`.
- **Calibration + score distribution** — the weekly monitoring DAG now computes reliability bins, Brier score, max calibration error, and 10-bucket probability distribution per run.
- **Segment-level drift** — PSI is now computed per segment (split by `latest_is_auto_renew` and derived `has_cancel_history`) in addition to feature-level drift.
- **Minimum-sample guard** — when the labelled current window has fewer than `MONITORING_MIN_SAMPLES` rows (default `1000`) the run is recorded with `status='insufficient_data'` and the retraining DAG is skipped. Prevents retraining on noisy, sparse windows.
- **Business-metric promotion gate** — `register_model.py` now requires both an AUC improvement AND no regression on a business metric (default `precision_churn`) beyond `BUSINESS_METRIC_TOLERANCE` (default `0.01`). Regressions produce the new decision `kept_existing_champion_business_regression`.

**One-time migration** (only needed on pre-existing databases — fresh `docker compose up` gets the schema from `db/init/`):

```bash
docker exec -i bt4301_postgres psql -U bt4301 -d kkbox \
  < db/migrations/add_model_monitoring_results.sql
```

**Run end-to-end** (DB + MLflow + Airflow stack must be up):

```bash
# 1. Score customers — writes model_version, threshold_version, feature_snapshot_id
python source/mlops/score_churn.py all

# 2. Register (runs the business-metric gate)
python source/mlops/register_model.py

# 3. Weekly monitoring (writes calibration, segment PSI, and sample counts)
docker exec bt4301_airflow_scheduler \
  airflow dags trigger weekly_model_monitoring

# 4. Retraining DAG reads monitoring status, skips on insufficient_data
docker exec bt4301_airflow_scheduler \
  airflow dags trigger automated_retraining
```

**Inspect results:**

```sql
-- Provenance on latest predictions
SELECT DISTINCT model_version, threshold_version, feature_snapshot_id
FROM processed.churn_predictions
ORDER BY 1, 2 NULLS LAST LIMIT 5;

-- Latest monitoring run
SELECT status, labeled_sample_count, min_sample_threshold,
       brier_score, max_calibration_error,
       jsonb_pretty(segment_psi) AS segment_psi
FROM processed.model_monitoring_results
ORDER BY monitored_at DESC LIMIT 1;

-- Registry evidence — new business-metric fields
cat docs/artifacts/model_registry.json | python -m json.tool
```

**Environment overrides:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `MONITORING_MIN_SAMPLES` | `1000` | Minimum labelled rows for an actionable verdict |
| `MAX_CALIBRATION_ERROR` | `0.10` | Max bin gap before flagging calibration drift |
| `SEGMENT_PSI` | `0.20` | PSI threshold per segment |
| `BUSINESS_METRIC_KEY` | `precision_churn` | Metric used in the promotion gate |
| `BUSINESS_METRIC_TOLERANCE` | `0.01` | Max absolute regression tolerated |
| `CLASSIFICATION_THRESHOLD_VERSION` | `v1` | Version tag written into every scored row |

**Run the unit tests:**

```bash
python -m pytest source/tests/test_mlops_monitoring.py \
                  source/tests/test_mlops_promotion_gate.py -v
```

Tests are pure numpy/pandas — no MLflow, Airflow, or Postgres dependencies at runtime.

---

## Airflow DAGs

Current DAGs:

| DAG name                         | Description                                           |
| -------------------------------- | ----------------------------------------------------- |
| `transform_and_track_lineage`    | Transform features + track lineage                    |
| `dataops_e2e_pipeline`           | Full DataOps chain (ingest → EDA report)              |
| `daily_churn_scoring`            | Daily scoring pipeline                                |
| `weekly_model_monitoring`        | Weekly drift + degradation monitoring                 |
| `automated_retraining`           | Retraining triggered when drift thresholds are breached |

### DAG Schedules

| DAG                            | Schedule     | How Triggered                                                                          |
| ------------------------------ | ------------ | -------------------------------------------------------------------------------------- |
| `transform_and_track_lineage`  | Manual       | Trigger manually in Airflow UI                                                         |
| `dataops_e2e_pipeline`         | Manual       | Trigger manually in Airflow UI                                                         |
| `daily_churn_scoring`          | `@daily`     | Auto-runs daily; can also be triggered manually                                        |
| `weekly_model_monitoring`      | `@weekly`    | Auto-runs weekly; triggers `automated_retraining` when PSI > 0.2 or AUC delta > 0.05  |
| `automated_retraining`         | Event-driven | Triggered automatically by `weekly_model_monitoring` when drift thresholds are breached |

### Task chains

`dataops_e2e_pipeline`:
`ingest_raw → cleanse → transform_features → watermark_data → validate_data → track_lineage → trigger_eda → generate_eda_images_report`

- **watermark_data** — computes a SHA-256 content hash of `processed.customer_features` and records it in `processed.data_watermarks` for data provenance and audit trail.
- **validate_data** — runs data quality checks (row count match, no nulls in key columns, value range validation, no duplicates). Fails the DAG if any critical check fails.

`daily_churn_scoring`:
`load_features -> load_production_model -> score -> write_predictions`

`weekly_model_monitoring`:
`compute_and_log_monitoring_metrics -> evaluate_thresholds -> (trigger_alert | no_alert_needed)`

### Option A — Run Airflow via Docker (recommended for both macOS and Windows)

This is the easiest approach and avoids platform compatibility issues.

**macOS/Linux:**

```bash
docker run --name airflow-us8 --rm -it -p 8080:8080 \
  -v "$(pwd):/opt/project" \
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/project/source/dataops/airflow/dags \
  -e PROJECT_ROOT=/opt/project \
  -e POSTGRES_HOST=host.docker.internal \
  -e POSTGRES_PORT=5432 \
  -e POSTGRES_DB=kkbox \
  -e POSTGRES_USER=bt4301 \
  -e POSTGRES_PASSWORD=bt4301pass \
  apache/airflow:2.9.3 \
  bash -lc "pip install psycopg2-binary pandas numpy matplotlib seaborn scikit-learn mlflow xgboost imbalanced-learn joblib && airflow standalone"
```

**Windows — PowerShell:**

```powershell
docker run --name airflow-us8 --rm -it -p 8080:8080 `
  -v "${PWD}:/opt/project" `
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/project/source/dataops/airflow/dags `
  -e PROJECT_ROOT=/opt/project `
  -e POSTGRES_HOST=host.docker.internal `
  -e POSTGRES_PORT=5432 `
  -e POSTGRES_DB=kkbox `
  -e POSTGRES_USER=bt4301 `
  -e POSTGRES_PASSWORD=bt4301pass `
  apache/airflow:2.9.3 `
  bash -lc "pip install psycopg2-binary pandas numpy matplotlib seaborn scikit-learn mlflow xgboost imbalanced-learn joblib && airflow standalone"
```

**Windows — CMD:**

```cmd
docker run --name airflow-us8 --rm -it -p 8080:8080 ^
  -v "%cd%:/opt/project" ^
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/project/source/dataops/airflow/dags ^
  -e PROJECT_ROOT=/opt/project ^
  -e POSTGRES_HOST=host.docker.internal ^
  -e POSTGRES_PORT=5432 ^
  -e POSTGRES_DB=kkbox ^
  -e POSTGRES_USER=bt4301 ^
  -e POSTGRES_PASSWORD=bt4301pass ^
  apache/airflow:2.9.3 ^
  bash -lc "pip install psycopg2-binary pandas numpy matplotlib seaborn scikit-learn mlflow xgboost imbalanced-learn joblib && airflow standalone"
```

Open [http://localhost:8080](http://localhost:8080), trigger the DAG you want to test, and verify all tasks are green. For the MLOps flow, the usual order is `daily_churn_scoring` followed by `weekly_model_monitoring`.

> **Windows note:** Native Airflow is not supported on Windows due to `fcntl` import errors. Use the Docker method above.

---

### Option B — Run Airflow natively (macOS only)

1. Install Airflow in a virtual environment:

```bash
 export AIRFLOW_HOME=~/airflow
 pip install "apache-airflow==2.9.3" \
   --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.10.txt"
```

2. Initialise the database and create an admin user:

```bash
 airflow db migrate
 airflow users create \
   --username admin --password admin \
   --firstname Admin --lastname User \
   --role Admin --email admin@example.com
```

3. Set environment variables and start Airflow:

```bash
 export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/source/dataops/airflow/dags
 export PROJECT_ROOT=$(pwd)
 export POSTGRES_HOST=127.0.0.1
 export POSTGRES_PORT=5432
 export POSTGRES_DB=kkbox
 export POSTGRES_USER=bt4301
 export POSTGRES_PASSWORD=bt4301pass
 airflow standalone
```

4. Open [http://localhost:8080](http://localhost:8080), trigger the DAG you want to test, and verify all tasks are green.

---

## Testing

### Unit / Smoke Tests (CI)

Run all non-integration tests (no Docker stack required):

```bash
pytest -q tests source/tests -m "not integration"
```

Focused Workstream C verification:

```bash
pytest -q source/tests/test_mlops_train_serve_consistency.py -m "not integration"
```

### End-to-End Integration Test (Local)

Requires the full Docker Compose stack running with a production model registered in MLflow:

```bash
docker compose up -d
pytest source/tests/test_e2e_integration.py -v -s
```

The test triggers the DataOps DAG, waits for completion, triggers the Scoring DAG, then queries the webapp `/customer/{id}/churn-risk` API and asserts a valid churn score is returned.
