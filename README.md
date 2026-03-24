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
│   └── processed/                # Processed/cleaned datasets
│
├── README.md
└── requirements.txt
```

**Note:** Remove `.gitkeep` files from `docs/`, `source/dataops/`, `source/mlops/`, `data/raw/`, and `data/processed/` before creating the final submission zip.

## Data Setup

The raw datasets (~1.9 GB) come from the [KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) on Kaggle and are too large to commit to Git. Use the download script to fetch them.

### Prerequisites

1. Install the Kaggle CLI: `pip install kaggle`
2. Set up your Kaggle API credentials:
   - Go to https://www.kaggle.com/settings -> **API** -> **Create New Token**
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
   - Run `chmod 600 ~/.kaggle/kaggle.json`
3. Accept the [competition rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules) (required for the download API to work)

### Download

```bash
bash data/download_data.sh
```

This downloads and unzips the following files into `data/raw/`:

| File                  | Description                                 |
| --------------------- | ------------------------------------------- |
| `train_v2.csv`        | Training labels — whether each user churned |
| `members_v3.csv`      | User demographic and registration info      |
| `transactions_v2.csv` | Payment transaction history                 |
| `user_logs_v2.csv`    | Daily listening activity logs               |

## Running the Pipeline

### 1. Start PostgreSQL

```bash
docker compose up -d
```

Starts a PostgreSQL 15 container (`bt4301_postgres`) and automatically creates all schemas and empty tables from `db/init/`. Wait for the health check:

```bash
docker ps  # STATUS should show "(healthy)"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set DB environment variables

PowerShell:

```powershell
$env:POSTGRES_HOST="127.0.0.1"
$env:POSTGRES_PORT="5432"
$env:POSTGRES_DB="kkbox"
$env:POSTGRES_USER="bt4301"
$env:POSTGRES_PASSWORD="bt4301pass"
```

CMD:

```cmd
set POSTGRES_HOST=127.0.0.1
set POSTGRES_PORT=5432
set POSTGRES_DB=kkbox
set POSTGRES_USER=bt4301
set POSTGRES_PASSWORD=bt4301pass
```

### 4. Run full script pipeline

```bash
python source/dataops/load_raw_data.py
python source/dataops/cleanse_data.py
python source/dataops/build_customer_features.py
python source/dataops/generate_lineage.py
python source/dataops/run_eda.py
python source/dataops/generate_eda_images_report.py
```

What each script does:

- `load_raw_data.py`: loads `data/raw/*.csv` into `raw.*`
- `cleanse_data.py`: cleansing/prep and writes `data/processed/df_train_final.csv`
- `build_customer_features.py`: refreshes `processed.customer_features`
- `generate_lineage.py`: refreshes `processed.data_lineage`
- `run_eda.py`: writes EDA outputs to `data/processed/eda/`
- `generate_eda_images_report.py`: generates 13 EDA charts and HTML report with those charts to `data/processed/eda/`

### 5. Feature selection + importance (MLflow)

This step performs **permutation importance** analysis and selects a final non-redundant feature set from the feature store. It also logs the resulting feature list as an **MLflow artifact**.

```bash
python source/mlops/feature_selection.py
```

Outputs are written to `docs/artifacts/` and logged to MLflow (local `mlruns/` by default).

### 5. Validate SQL outputs

```sql
SELECT COUNT(*) AS raw_train_rows FROM raw.train;
SELECT COUNT(*) AS customer_feature_rows FROM processed.customer_features;
SELECT COUNT(*) AS lineage_rows FROM processed.data_lineage;

WITH feature_cols AS (
  SELECT column_name
  FROM information_schema.columns
  WHERE table_schema='processed'
    AND table_name='customer_features'
    AND column_name <> 'msno'
),
lineage_feats AS (
  SELECT DISTINCT feature_name
  FROM processed.data_lineage
)
SELECT
  (SELECT COUNT(*)
   FROM feature_cols fc
   LEFT JOIN lineage_feats lf ON fc.column_name=lf.feature_name
   WHERE lf.feature_name IS NULL) AS missing_in_lineage,
  (SELECT COUNT(*)
   FROM lineage_feats lf
   LEFT JOIN feature_cols fc ON lf.feature_name=fc.column_name
   WHERE fc.column_name IS NULL) AS extra_in_lineage;
```

Expected:

- `raw_train_rows == customer_feature_rows`
- `missing_in_lineage = 0`
- `extra_in_lineage = 0`

### 6. Airflow DAGs

Current DAGs:

- `us6_transform_and_track_lineage`
- `us8_dataops_e2e_pipeline`

US-08 chain:
`ingest_raw -> cleanse -> transform_features -> track_lineage -> trigger_eda -> generate_eda_images_report`

### 7. Run Airflow on Windows (recommended via Docker)

Native Windows Airflow may fail due `fcntl` import errors. Use Docker Airflow:

```cmd
docker run --name airflow-us8 --rm -it -p 8080:8080 ^
  -v "%cd%:/opt/project" ^
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/project/source/dataops/airflow/dags ^
  -e PROJECT_ROOT=/opt/project ^
  -e POSTGRES_HOST=host.docker.internal ^
  -e POSTGRES_PORT=5433 ^
  -e POSTGRES_DB=kkbox ^
  -e POSTGRES_USER=bt4301 ^
  -e POSTGRES_PASSWORD=bt4301pass ^
  apache/airflow:2.9.3 ^
  bash -lc "pip install psycopg2-binary pandas numpy matplotlib seaborn && airflow standalone"
```

Open `http://localhost:8080`, trigger `us8_dataops_e2e_pipeline`, and verify all tasks are green.

### 8. Run Airflow on macOS (Docker)

Use Docker Airflow similarly on macOS:

```bash
docker run --name airflow-us8 --rm -it -p 8080:8080 \
  -v "$(pwd):/opt/project" \
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/project/source/dataops/airflow/dags \
  -e PROJECT_ROOT=/opt/project \
  -e POSTGRES_HOST=host.docker.internal \
  -e POSTGRES_PORT=5433 \
  -e POSTGRES_DB=kkbox \
  -e POSTGRES_USER=bt4301 \
  -e POSTGRES_PASSWORD=bt4301pass \
  apache/airflow:2.9.3 \
  bash -lc "pip install psycopg2-binary pandas numpy matplotlib seaborn && airflow standalone"
```

Open `http://localhost:8080`, trigger `us8_dataops_e2e_pipeline`, and verify all tasks are green.

### 9. Run Airflow on macOS (native)

1. Install Airflow in a virtual environment:

```bash
export AIRFLOW_HOME=~/airflow
pip install "apache-airflow==2.9.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.10.txt"
```

2. Initialise the database and create an admin user:

```bash
airflow db migrate
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin --email admin@example.com
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

4. Open `http://localhost:8080`, trigger `us8_dataops_e2e_pipeline`, and verify all tasks are green.
