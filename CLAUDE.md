# CLAUDE.md

## Project Overview

BT4301 Group 9 — **KKBox Churn Prediction Platform**. End-to-end DataOps + MLOps + DevOps pipeline predicting music streaming subscriber churn. Data source: KKBox Churn Prediction Challenge (Kaggle), ~1.9 GB across 4 CSVs.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Database | PostgreSQL 15 (Docker) |
| Data Processing | Python 3.10+, pandas, psycopg2 |
| Orchestration | Apache Airflow |
| Experiment Tracking | MLflow (planned) |
| Web App | Flask or FastAPI (planned) |
| CI/CD | GitHub Actions (planned) |
| Containerisation | Docker Compose |

## Directory Structure

```
├── db/init/                        # SQL schema files (run automatically by Docker)
│   ├── 00_schemas.sql             #   raw, processed, predictions, lineage schemas
│   ├── 01_raw_tables.sql          #   raw.train, raw.members, raw.transactions, raw.user_logs
│   └── 02_processed.sql           #   processed.customer_features, processed.data_lineage
│
├── source/
│   ├── dataops/                   # DataOps scripts
│   │   ├── load_raw_data.py      #   Bulk CSV → raw.* via COPY
│   │   ├── build_customer_features.py  # raw.* → processed.customer_features
│   │   ├── feature_registry.py   #   FeatureSpec dataclass registry (21 features)
│   │   ├── generate_lineage.py   #   Populate processed.data_lineage
│   │   ├── sample_dataset.py     #   20% stratified sample for dev
│   │   ├── data_cleaning_eda.ipynb  # EDA and cleansing notebook
│   │   └── airflow/dags/
│   │       └── transform_and_lineage_dag.py
│   └── mlops/                     # MLOps scripts (placeholder)
│
├── data/raw/                      # Raw CSVs (git-ignored)
├── data/processed/                # Processed outputs (git-ignored)
├── config/cleansing_rules.yaml    # Data cleansing rules per table
├── docs/                          # Documentation
│   ├── final-product-overview.md  # Full product spec from all 29 user stories
│   └── dataset-reduction.md       # Sampling methodology
├── .env                           # PostgreSQL credentials
├── docker-compose.yml             # PostgreSQL 15 container
└── requirements.txt               # Python dependencies
```

## Database

**Schemas:** `raw`, `processed`, `predictions`, `lineage`

**Key tables:**
- `raw.train` — churn labels (msno, is_churn)
- `raw.members` — demographics (city, bd, gender, registered_via)
- `raw.transactions` — payment history (payment_method_id, plan_list_price, is_auto_renew, etc.)
- `raw.user_logs` — daily listening activity (num_25/50/75/985/100, num_unq, total_secs)
- `processed.customer_features` — one row per customer, 21 features + target
- `processed.data_lineage` — feature-level lineage metadata

**Connection pattern** (used in all scripts):
```python
DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST",     "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB",       "kkbox"),
    "user":     os.getenv("POSTGRES_USER",     "bt4301"),
    "password": os.getenv("POSTGRES_PASSWORD", "bt4301pass"),
}
```

## Coding Conventions

- **Always** start files with `from __future__ import annotations`
- **DB connections:** `psycopg2.connect(**DB_CONFIG)` with `autocommit = False`, try/except/finally, `conn.rollback()` on error, close cursor and connection in finally
- **SQL:** store as module-level string constants (not inline in functions)
- **Constants:** `SCREAMING_SNAKE_CASE` (e.g., `DATA_DIR`, `CHUNK_SIZE`, `TABLES`)
- **File paths:** compute from `os.path.dirname(__file__)`, never hardcode absolute paths
- **Module structure:** docstring → imports (`__future__` → stdlib → third-party → local) → constants → helpers → `main()` → `if __name__ == "__main__"` guard
- **Scripts must work standalone AND via Airflow** — no Airflow-specific imports in core logic
- **Validation:** add validation functions for critical transformations (row counts, NULL checks, coverage percentages, churn rate preservation)

## Feature Registry

`source/dataops/feature_registry.py` — single source of truth for all 21 features:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    source_table: str
    transformation_rule: str

FEATURE_SPECS = [FeatureSpec(...), ...]
```
Import as: `from feature_registry import FEATURE_SPECS`

## User Stories

29 user stories tracked in [GitHub Project #3](https://github.com/users/yongggquannn/projects/3):
