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

### 1. Start the database

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

### 3. Load raw data into PostgreSQL

```bash
python source/dataops/load_raw_data.py
```

Bulk-loads all four CSV files from `data/raw/` into the `raw.*` tables. Expected output:

```
[raw.train]        Loaded: 194,192 rows
[raw.members]      Loaded: 172,033 rows
[raw.transactions] Loaded: 226,143 rows
[raw.user_logs]    Loaded: 2,705,095 rows
```

> Re-running is safe — each table is truncated before loading.

### 4. Build the feature table

```bash
python source/dataops/build_customer_features.py
```

Joins and aggregates the four raw tables into `processed.customer_features` — one row per customer with demographic, contract, usage, and churn label features.

### 5. Verify the results

Connect with any PostgreSQL client (TablePlus, psql, pgAdmin):

| Field    | Value        |
| -------- | ------------ |
| Host     | `localhost`  |
| Port     | `5432`       |
| Database | `kkbox`      |
| User     | `bt4301`     |
| Password | `bt4301pass` |

Quick sanity queries:

```sql
-- Row count should match raw.train
SELECT COUNT(*) FROM processed.customer_features;

-- Churn distribution
SELECT is_churn, COUNT(*) FROM processed.customer_features GROUP BY is_churn;

-- Sample rows
SELECT * FROM processed.customer_features LIMIT 5;
```

### 6. Run the Airflow DAG (US-6 — transform + lineage)

> Prerequisites: steps 1–4 must be complete (database up, raw data loaded).

Install Airflow (one-time):

```bash
pip install apache-airflow
airflow db migrate # initialise the Airflow metadata DB
```

Point Airflow at the project DAGs folder:

```bash
export AIRFLOW**CORE**DAGS_FOLDER=$(pwd)/source/dataops/airflow/dags
```

Start the scheduler and web server (two separate terminals, or background):

```bash
airflow scheduler &
airflow webserver --port 8080 &
```

Trigger the DAG from the UI at http://localhost:8080, or via CLI:

```bash
airflow dags trigger us6_transform_and_track_lineage
```

Verify lineage records were written to PostgreSQL:

```sql
SELECT * FROM lineage.data_lineage ORDER BY created_at DESC LIMIT 10;
```

Expected: 21 rows — one per feature column in `processed.customer_features`.

To reset and re-run the full pipeline from scratch:

```bash
docker compose down -v   # wipes the Postgres volume
docker compose up -d
python source/dataops/load_raw_data.py
python source/dataops/build_customer_features.py
```
