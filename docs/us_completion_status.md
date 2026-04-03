# User Story Completion Status

> **Last updated:** 2026-03-31
> **Canonical backlog:** [GitHub Project #3](https://github.com/users/yongggquannn/projects/3)

---

## Completed Stories (Sprint 1 & 2)

### US-01 — PostgreSQL Schema Setup

| Acceptance Criteria                                                               | Status                |
| --------------------------------------------------------------------------------- | --------------------- |
| PostgreSQL instance running (not SQLite)                                          | Done                  |
| `raw` schema created with tables: `members`, `transactions`, `user_logs`, `train` | Done                  |
| `processed` schema created as placeholder for feature table                       | Done                  |
| Screenshot of pgAdmin or psql showing both schemas                                | **Screenshot needed** |

### US-02 — Raw Data Ingestion

| Acceptance Criteria                                 | Status |
| --------------------------------------------------- | ------ |
| All 4 KKBox CSVs loaded into raw schema             | Done   |
| Row counts verified against source files and logged | Done   |
| Python ingest script committed to GitHub            | Done   |

### US-03 — Data Cleansing

| Acceptance Criteria                                                           | Status           |
| ----------------------------------------------------------------------------- | ---------------- |
| Null handling strategy documented per column                                  | Done             |
| Type casting applied (dates, numeric, categorical)                            | Done             |
| Duplicate records removed with count logged                                   | Done             |
| Cleansing rules stored as a config file (e.g., `config/cleansing_rules.yaml`) | **Missing file** |

### US-04 — Feature Table Construction

| Acceptance Criteria                                                                 | Status |
| ----------------------------------------------------------------------------------- | ------ |
| `processed.customer_features` table created with one row per customer               | Done   |
| Includes: demographic, contract, usage aggregates, transaction history, churn label | Done   |
| Row count matches training set                                                      | Done   |

### US-05 — Data Watermarking

| Acceptance Criteria                                  | Status |
| ---------------------------------------------------- | ------ |
| `ingestion_timestamp` column added to all raw tables | Done   |
| Populated automatically on load                      | Done   |

### US-06 — Data Lineage Tracking

| Acceptance Criteria                                                                                                       | Status |
| ------------------------------------------------------------------------------------------------------------------------- | ------ |
| `processed.data_lineage` metadata table with columns: `feature_name`, `source_table`, `transformation_rule`, `created_at` | Done   |
| Populated for all features in `customer_features`                                                                         | Done   |
| Python script or Airflow task generating lineage records                                                                  | Done   |

### US-07 — Exploratory Data Analysis

| Acceptance Criteria                                                                                                                 | Status |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------ |
| Jupyter notebook committed to `source/` with churn rate analysis, distribution plots, correlation heatmap, class imbalance analysis | Done   |

### US-08 — Airflow DataOps DAG

| Acceptance Criteria                                                                               | Status |
| ------------------------------------------------------------------------------------------------- | ------ |
| Airflow DAG with tasks: `ingest_raw → cleanse → transform_features → track_lineage → trigger_eda` | Done   |
| All tasks succeed (green) in Airflow UI                                                           | Done   |

### US-09 — MLflow Tracking Server Setup

| Acceptance Criteria                                                      | Status                |
| ------------------------------------------------------------------------ | --------------------- |
| MLflow tracking server running, connected to PostgreSQL as backend store | Done                  |
| Artifact store configured (local path or S3-compatible)                  | Done                  |
| MLflow UI accessible, screenshot of experiment list                      | **Screenshot needed** |

### US-10 — Model Training and Comparison

| Acceptance Criteria                                                              | Status                |
| -------------------------------------------------------------------------------- | --------------------- |
| 3 MLflow runs under experiment `KKBox Churn`                                     | Done                  |
| Each run logs: hyperparameters, AUC-ROC, F1, precision, recall, confusion matrix | Done                  |
| Feature importance plot logged as artifact for tree-based model                  | Done                  |
| Run comparison screenshot from MLflow UI                                         | **Screenshot needed** |

### US-11 — Feature Engineering and Selection

| Acceptance Criteria                                            | Status |
| -------------------------------------------------------------- | ------ |
| Feature importance / permutation importance analysis completed | Done   |
| Final feature set documented (count, selection method used)    | Done   |
| Feature list logged as MLflow artifact                         | Done   |

### US-12 — MLflow Model Registry

| Acceptance Criteria                                                  | Status                |
| -------------------------------------------------------------------- | --------------------- |
| Best model registered as `KKBox-Churn-Classifier` in MLflow registry | Done                  |
| Version history showing: None → Staging → Production transitions     | Done                  |
| Screenshot of registry with version and stage labels                 | **Screenshot needed** |

### US-13 — Airflow Scoring DAG

| Acceptance Criteria                                                                                    | Status                |
| ------------------------------------------------------------------------------------------------------ | --------------------- |
| Airflow DAG: `load_features → load_production_model → score → write_predictions`                       | Done                  |
| `processed.churn_predictions` table with: `customer_id`, `churn_probability`, `risk_tier`, `scored_at` | Done                  |
| Screenshot of successful DAG run and sample rows from predictions table                                | **Screenshot needed** |

### US-14 — Airflow Monitoring DAG

| Acceptance Criteria                                             | Status                |
| --------------------------------------------------------------- | --------------------- |
| Monitoring DAG computes PSI for top 5 features                  | Done                  |
| Compares current AUC against baseline, logs delta to PostgreSQL | Done                  |
| If PSI > 0.2 or AUC delta > 0.05 → Airflow alert task triggers  | Done                  |
| Screenshot of monitoring DAG and sample monitoring results      | **Screenshot needed** |

### US-15 — Model Governance and Model Card

| Acceptance Criteria                                                                                     | Status |
| ------------------------------------------------------------------------------------------------------- | ------ |
| Model card stored as MLflow artifact (training data, features, metrics, limitations, fairness analysis) | Done   |

### US-16 — Minimal Web App v1

| Acceptance Criteria                                           | Status                |
| ------------------------------------------------------------- | --------------------- |
| FastAPI endpoint: `GET /customer/{customer_id}/churn-risk`    | Done                  |
| Returns: `churn_probability`, `risk_tier`, `top_3_features`   | Done                  |
| Simple HTML page with customer lookup form and result display | Done                  |
| Screenshot of working prototype                               | **Screenshot needed** |

### US-18 — Class Imbalance Handling

| Acceptance Criteria                                                 | Status |
| ------------------------------------------------------------------- | ------ |
| Two MLflow runs: one with SMOTE, one with `class_weight="balanced"` | Done   |
| Precision/recall/F1 comparison table showing trade-offs             | Done   |
| Chosen strategy documented with justification                       | Done   |

### US-22 — Misclassification Analysis

| Acceptance Criteria                                                         | Status |
| --------------------------------------------------------------------------- | ------ |
| Confusion matrix breakdown: FN vs FP counts                                 | Done   |
| Feature value distributions compared: misclassified vs correctly classified | Done   |
| At least 2 actionable insights documented in report                         | Done   |

---

## In Progress

### US-20 — Champion-Challenger Framework

| Acceptance Criteria                                                                    | Status                |
| -------------------------------------------------------------------------------------- | --------------------- |
| Python function: if `new_model.auc > champion.auc + threshold` → promote to production | Done                  |
| New model version visible in MLflow registry with promotion audit trail                | Done                  |
| Screenshot of registry showing version history across Sprint 2 and Sprint 3            | **Screenshot needed** |

### US-21 — Automated Retraining DAG

| Acceptance Criteria                                                          | Status                |
| ---------------------------------------------------------------------------- | --------------------- |
| Airflow DAG: `check_drift_results → retrain_if_needed → evaluate → register` | Done                  |
| DAG triggers full retrain only if PSI > 0.2 or AUC delta > 0.05 (from US-14) | Done                  |
| Screenshot of DAG with conditional branch logic                              | **Screenshot needed** |

### US-23 — Complete Churn Risk Web App

| Acceptance Criteria                                                                  | Status |
| ------------------------------------------------------------------------------------ | ------ |
| `/` — Search bar: input customer ID, submit button                                   | Todo   |
| `/customer/<id>` — Churn probability (%), risk tier badge, top 3 SHAP features       | Todo   |
| `/dashboard` — Table of top 50 highest-risk customers, sortable by churn probability | Todo   |
| Reads live from `processed.churn_predictions` in PostgreSQL                          | Todo   |
| Screenshot of all three pages                                                        | Todo   |

### US-24 — GitHub Actions CI/CD Pipeline

| Acceptance Criteria                                              | Status |
| ---------------------------------------------------------------- | ------ |
| `.github/workflows/deploy.yml` committed to repo                 | Todo   |
| Workflow triggers on push to `main` and on model promotion event | Todo   |
| Pipeline steps: lint → test → build Docker image → deploy        | Todo   |
| Screenshot of successful Actions run                             | Todo   |

### US-25 — Docker Compose Full Stack

| Acceptance Criteria                                                        | Status |
| -------------------------------------------------------------------------- | ------ |
| `docker-compose.yml` spinning up: PostgreSQL, Airflow, MLflow, FastAPI app | Todo   |
| `docker compose up` from a clean machine runs the full stack               | Todo   |
| `README.md` with setup instructions                                        | Todo   |

### US-26 — End-to-End Integration Testing

| Acceptance Criteria                        | Status |
| ------------------------------------------ | ------ |
| Integration test script in `source/tests/` | Todo   |
| All assertions pass                        | Todo   |
| Screenshot of test output                  | Todo   |

---

## Deliverables

### US-27 — Project Report (due 19 April)

| Acceptance Criteria                                                   | Status |
| --------------------------------------------------------------------- | ------ |
| Report covers all 10 sections from the project spec                   | Todo   |
| All screenshots, source code references, and Kanban evidence included | Todo   |
| Formatted per spec: A4, Times New Roman 12pt, double spacing, DOCX    | Todo   |
| All group members reviewed and signed off                             | Todo   |

### US-28 — Peer Learning Presentation Slides (due 16 April)

| Acceptance Criteria                                                               | Status |
| --------------------------------------------------------------------------------- | ------ |
| Slide deck covers: problem → solution → DataOps → MLOps → DevOps → business value | Todo   |
| Fits within 12-minute slot                                                        | Todo   |
| All members have a speaking role assigned                                         | Todo   |

### US-29 — Full Demonstration Video (due 21 April)

| Acceptance Criteria                                                           | Status |
| ----------------------------------------------------------------------------- | ------ |
| Video covers: Airflow DataOps, MLflow, Scoring + Monitoring DAG, Web App demo | Todo   |
| Video is <= 30 minutes                                                        | Todo   |

---

## Sprint 3 Prioritization

### Dependency Graph

```
US-10 / US-12 / US-18 ──→ US-20 (Champion-Challenger)
US-19 (SHAP)   ──→ US-23 (Full Web App)
US-11 / US-14 / US-18 / US-20 ──→ US-21 (Retraining DAG)
US-25 (Docker)  ──→ US-29 (Demo Video)
```

### Suggested Weekly Plan

| Week                        | Stories                           | Focus                                    |
| --------------------------- | --------------------------------- | ---------------------------------------- |
| **Week 1** (31 Mar – 6 Apr) | US-17, US-19, US-20               | Model improvements + SHAP explainability |
| **Week 2** (7 – 13 Apr)     | US-21, US-23, US-25               | Pipeline completion + Docker full stack  |
| **Week 3** (14 – 21 Apr)    | US-24, US-26, US-28, US-27, US-29 | DevOps, testing + all deliverables       |

### Open Items (Screenshots / Minor Gaps)

- **US-01, 09, 10, 12, 13, 14, 16, 20, 21**: Screenshots needed from running services
- **US-03**: `config/cleansing_rules.yaml` file to be created
