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
│   ├── dataops/                 # DataOps sprint (.py, .ipynb)
│   ├── mlops/                   # MLOps sprint (.py, .ipynb)
│   └── (root-level scripts)
│
├── data/                    # Submission: data subfolder
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed/cleaned datasets
│
├── README.md
└── requirements.txt
```

**Note:** Remove `.gitkeep` files from `docs/`, `source/dataops/`, `source/mlops/`, `data/raw/`, and `data/processed/` before creating the final submission zip.

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

## Running the Pipeline

### Step 1 — Start PostgreSQL

```bash
docker compose up -d
```

This starts a PostgreSQL 15 container (`bt4301_postgres`) and automatically creates all schemas and tables from `db/init/`. Wait until the container is healthy:

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


---

### Step 5 — Feature selection and importance (MLflow)

Runs permutation importance analysis, selects a final non-redundant feature set, and logs the results to MLflow.

```bash
python source/mlops/feature_selection.py
```

Outputs are written to `docs/artifacts/` and logged to MLflow (local `mlruns/` directory by default).

---

### Step 6 — Handle class imbalance (US-18)

Runs two MLflow experiments comparing SMOTE oversampling vs. `class_weight="balanced"` model training.

```bash
python source/mlops/train_us18_class_imbalance.py
```

Evidence artifacts are written to `docs/artifacts/` and logged to MLflow (including `us18_chosen_strategy.json` for the next step).

---

### Step 7 — Train and compare models (US-10)

Trains Logistic Regression, XGBoost, and MLP on `processed.customer_features` using the feature list from Step 5 and (if present) the imbalance strategy from Step 6. Logs metrics and plots to MLflow; writes comparison artifacts to `docs/artifacts/` (e.g. `us10_model_comparison.csv`, `us10_best_model.json`).

**Prerequisites:** Steps 4–6 recommended (Step 6 can be skipped—training falls back to a default imbalance strategy if `us18_chosen_strategy.json` is missing).

```bash
python source/mlops/train_model.py
```

Quick smoke test (one model, subsampled rows):

```bash
python source/mlops/train_model.py --models logistic_regression --sample-rows 5000
```

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

- `us17_best_hyperparams.json` — best params and AUC-ROC
- `us17_optimization_curve.png` — Optuna optimisation history
- `us17_param_importance.png` — hyperparameter importance
- `us17_improvement_summary.md` — baseline vs tuned comparison

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

- `docs/us22_misclassification_analysis.md`
- `docs/artifacts/us22_confusion_breakdown.json`
- `docs/artifacts/us22_misclassified_cases.csv`
- `docs/artifacts/us22_numeric_distribution_comparison.csv`
- `docs/artifacts/us22_categorical_distribution_comparison.csv`
- `docs/artifacts/us22_feature_distribution_plot.png`

---

### Step 10 — Register best model in MLflow Model Registry (US-12)

Registers the best model (from US-10 training) into the MLflow Model Registry as `KKBox-Churn-Classifier` and promotes it through None → Staging → Production.

**Prerequisites:** The MLflow tracking server must be running. Choose one of the options below:

**Option A — Docker (recommended, works on all platforms):**

```bash
docker compose up
```

This starts both PostgreSQL and MLflow together. MLflow will be available at [http://localhost:5001](http://localhost:5001).

**Option B — macOS / Linux (local PostgreSQL required):**

1. Set your system username in `.env`:
  ```
   MLFLOW_POSTGRES_USER=<output of `whoami`>
  ```
2. Start the server:
  ```bash
   bash mlflow_server.sh
  ```

**Option C — Windows (local PostgreSQL required):**

1. Set your PostgreSQL credentials in `.env`:
  ```
   MLFLOW_POSTGRES_USER=postgres
   MLFLOW_POSTGRES_PASSWORD=yourpassword
  ```
2. Start the server:
  ```powershell
   .\mlflow_server.ps1
  ```

Then, in a separate terminal:

```bash
clear   # macOS/Linux
# $env:MLFLOW_TRACKING_URI="http://localhost:5001" # Windows PowerShell
python source/mlops/register_model.py
```

This registers `KKBox-Churn-Classifier` version 1 and transitions it to Production. Evidence is saved to `docs/artifacts/us12_model_registry.json`. Visit the MLflow UI at [http://localhost:5001/#/models/KKBox-Churn-Classifier](http://localhost:5001/#/models/KKBox-Churn-Classifier) to view the registry.

---

### Step 11 — Generate daily predictions from the production model (US-13)

Runs the scoring pipeline against the production model. The scoring script tries the MLflow registry model `models:/KKBox-Churn-Classifier/Production` when `MLFLOW_MODEL_URI` is not set, and only falls back to a small local model if the production model is unavailable.

```bash
python source/mlops/score_churn.py all
```

This writes predictions into `processed.churn_predictions`, which are then used by the monitoring DAG.

---

### Step 12 — Churn risk web app (US-16)

A minimal FastAPI web app that lets you look up a customer's churn risk.

**Prerequisites:** Steps 1–11 (PostgreSQL running, `processed.churn_predictions` populated).

**Install additional dependencies:**

```bash
pip install fastapi "uvicorn[standard]" jinja2
```

**Start the web app:**

```bash
python -m uvicorn source.webapp.app:app --host 0.0.0.0 --port 8000 --reload
```

**Verify:**

- **Browser:** Open [http://localhost:8000](http://localhost:8000), enter a customer ID, and click "Look up".
- **API:** `curl http://localhost:8000/customer/<customer_id>/churn-risk`

Returns `churn_probability` (float), `risk_tier` (High/Medium/Low), and `top_3_features` (global permutation importance).

---

### Step 13 — SHAP explainability (US-19)

Generates SHAP values for the production churn model, producing a summary plot (global feature importance) and waterfall plots for three sample customers (high/medium/low risk). SHAP values are also stored per-prediction in the database and logged to MLflow.

**Prerequisites:** Steps 1–11 (PostgreSQL running, `processed.churn_predictions` populated, model trained and registered).

**If upgrading an existing database** (table already exists without the `shap_values` column), run the migration first:

```bash
docker exec -i bt4301_postgres psql -U bt4301 -d kkbox < db/migrations/us19_add_shap_values.sql
```

**Run the SHAP script:**

```bash
python source/mlops/explain_shap.py
```

**CLI options:**

| Flag | Description |
| ---- | ----------- |
| `--tracking-uri` | MLflow tracking URI (default: local `mlruns/`) |
| `--top-k N` | Number of top SHAP features to store per prediction (default: 5) |
| `--skip-db-update` | Skip writing SHAP values back to the database |

**Artifacts produced (in `docs/artifacts/`):**

- `us19_shap_summary.png` — SHAP beeswarm summary plot
- `us19_shap_waterfall_high.png` — Waterfall plot for a high-risk customer
- `us19_shap_waterfall_medium.png` — Waterfall plot for a medium-risk customer
- `us19_shap_waterfall_low.png` — Waterfall plot for a low-risk customer

**Verify DB update:**

```sql
SELECT customer_id, shap_values
FROM processed.churn_predictions
WHERE shap_values IS NOT NULL
LIMIT 3;
```

---

### Step 14 — Model governance artifacts (US-15)

The production training flow in `source/mlops/train_model.py` now writes model-governance artifacts for the best model and logs them to MLflow.

Expected local artifacts in `docs/artifacts/`:

- `us15_model_card.md`
- `us15_feature_list_types.csv`
- `us15_fairness_gender.csv`
- `us15_fairness_age_band.csv`

Governance evidence guide:

- `docs/us15_governance_evidence.md`

---

### Step 15 — Weekly monitoring checks (US-14)

Run `US-15` first so the best model artifacts are ready, then continue with registration, scoring, and weekly monitoring.

The weekly monitoring DAG checks feature drift and model degradation.

It:

- computes PSI for the top 5 monitorable features
- compares current AUC against a baseline AUC
- logs results into `processed.model_monitoring_results`
- triggers an alert task if `PSI > 0.2` or `AUC delta > 0.05`

After triggering `daily_churn_scoring` and `us14_weekly_model_monitoring` in Airflow, connect to PostgreSQL and run:

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

- `docs/us14_monitoring_evidence.md`

---

### Step 16 — Validate database outputs

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

---

## Airflow DAGs

Current DAGs:


| DAG name                          | Description                              |
| --------------------------------- | ---------------------------------------- |
| `us6_transform_and_track_lineage` | Transform features + track lineage       |
| `us8_dataops_e2e_pipeline`        | Full DataOps chain (ingest → EDA report) |
| `daily_churn_scoring`             | Daily scoring pipeline (US-13)           |
| `us14_weekly_model_monitoring`    | Weekly drift + degradation monitoring    |


US-08 task chain:
`ingest_raw → cleanse → transform_features → track_lineage → trigger_eda → generate_eda_images_report`

### Option A — Run Airflow via Docker (recommended for both macOS and Windows)

US-13 task chain:
`load_features -> load_production_model -> score -> write_predictions`

US-14 task chain:
`compute_and_log_monitoring_metrics -> evaluate_thresholds -> (trigger_alert | no_alert_needed)`

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

Open [http://localhost:8080](http://localhost:8080), trigger the DAG you want to test, and verify all tasks are green. For the MLOps flow, the usual order is `daily_churn_scoring` followed by `us14_weekly_model_monitoring`.

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

