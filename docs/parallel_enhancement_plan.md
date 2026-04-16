# Parallel Enhancement Plan

Execution plan for parallel work across the app, DataOps, and MLOps layers.

## Goal

Upgrade the project from a technically complete churn pipeline into a stronger product:

- the app should recommend actions, not just show scores
- the platform should express business value, not just model metrics
- the pipeline should be reproducible, observable, and production-safe

## Priority Order

1. Product-facing app improvements that change the demo narrative
2. MLOps fixes that remove train/serve risk
3. DataOps improvements that improve reproducibility and trust
4. Monitoring and governance enhancements

## Parallel Workstreams

## Workstream A — App: Actionable Retention Product

### Objective

Turn the web app from a lookup dashboard into an action-oriented retention tool.

### Tasks

- Add recommended next actions on the customer detail page
- Add business-friendly explanation text alongside SHAP output
- Add dashboard summary cards for risk counts and churn distribution
- Add prioritization fields such as customer value proxy and intervention priority
- Add filters for risk tier and customer segment on the dashboard

### Deliverables

- Customer page with action recommendation and explanation
- Dashboard with summary metrics and prioritization
- Updated screenshots for presentation and report

### Suggested File Targets

- `source/webapp/app.py`
- `source/webapp/templates/index.html`
- `source/webapp/templates/dashboard.html`
- `source/webapp/templates/customer.html`

### Notes

- Keep the UI focused on business users
- Translate technical explanations into plain English where possible
- Prefer simple heuristics for action recommendations if no policy engine exists yet

## Workstream B — App: Manager and Operations View

### Objective

Add a manager-facing layer that shows portfolio-level churn risk and system health.

### Tasks

- Add summary cards for high-risk customers, medium-risk customers, and total scored customers
- Add trend or distribution view for churn probabilities
- Add segment breakdowns such as tenure, payment behavior, or usage intensity
- Add model freshness and latest scoring timestamp to the dashboard
- Add monitoring status indicators if recent drift results are available

### Deliverables

- Expanded dashboard for managerial review
- Clear story for “what happened this week” and “where to intervene”

### Suggested File Targets

- `source/webapp/app.py`
- `source/webapp/templates/dashboard.html`

### Notes

- This workstream can run in parallel with Workstream A if template ownership is coordinated
- If multiple people work on the same template, split ownership by route or section

## Workstream C — MLOps: Train/Serve Consistency

### Objective

Eliminate manual preprocessing reconstruction and ensure the registered model is the same artifact used in production scoring.

### Tasks

- Refactor training to log the full preprocessing-plus-model pipeline to MLflow
- Persist feature ordering and schema as part of the model artifact or signature
- Refactor scoring to load and use the registered pipeline directly
- Remove manual preprocessing duplication from scoring
- Ensure scoring fails clearly if the production model is unavailable

### Deliverables

- Single MLflow model artifact that is valid for both evaluation and scoring
- Simplified and safer scoring flow

### Suggested File Targets

- `source/mlops/train_model.py`
- `source/mlops/score_churn.py`
- `source/mlops/register_model.py`

### Notes

- This is the highest-priority technical refactor
- Do not keep a hidden fallback path that silently trains another model during inference

## Workstream D — DataOps: Feature Versioning and Lineage

### Objective

Make feature generation reproducible and traceable across training and scoring runs.

### Tasks

- Add a feature snapshot identifier or feature run identifier
- Record build timestamp, source watermark, and pipeline run metadata with feature builds
- Persist feature snapshot metadata in a dedicated table
- Ensure downstream training and scoring artifacts record which feature snapshot they used
- Expand watermark coverage beyond `msno` and `is_churn`

### Deliverables

- Queryable linkage between raw ingestion, feature generation, training, and scoring
- Reproducible feature-store history

### Suggested File Targets

- `source/dataops/build_customer_features.py`
- `source/dataops/watermark_data.py`
- `source/dataops/generate_lineage.py`
- `db/init/02_processed.sql`
- `source/common/db.py`

### Notes

- Keep the implementation lightweight if schema changes are needed
- A metadata table is sufficient if full snapshot tables are too expensive

## Workstream E — DataOps: Stronger Validation and Contracts

### Objective

Catch bad inputs and suspicious transformations before they affect model training or scoring.

### Tasks

- Add schema validation for critical tables and columns
- Add freshness checks for raw data
- Add coverage thresholds for key features
- Add distribution checks against prior runs or baselines
- Separate validation failures into warning vs blocking classes

### Deliverables

- More reliable validation stage in the DataOps DAG
- Better operator visibility into why a run failed

### Suggested File Targets

- `source/dataops/validate_data.py`
- `source/dataops/airflow/dags/dataops_e2e_dag.py`
- `docs/cleansing_rules.yaml`

### Notes

- Prefer small, explicit rules over a large framework
- Log validation results to a table if possible instead of only printing them

## Workstream F — MLOps: Monitoring and Promotion Quality

### Objective

Improve model governance so retraining and promotion decisions are more defensible.

### Tasks

- Add calibration or score distribution monitoring
- Add segment-level drift monitoring
- Add minimum sample-size guards before AUC-based monitoring decisions
- Compare champion and challenger using business metrics in addition to AUC
- Record model version, feature snapshot, and threshold version in scoring outputs

### Deliverables

- Richer monitoring evidence
- Stronger champion-challenger logic

### Suggested File Targets

- `source/dataops/airflow/dags/weekly_model_monitoring_dag.py`
- `source/dataops/airflow/dags/automated_retraining_dag.py`
- `source/mlops/register_model.py`
- `source/mlops/train_model.py`

### Notes

- This should follow Workstream C because better model packaging simplifies monitoring consistency

## Recommended Execution Sequence

### Phase 1 — Demo Impact

- Workstream A
- Workstream B

### Phase 2 — Technical Credibility

- Workstream C
- Workstream D

### Phase 3 — Operational Maturity

- Workstream E
- Workstream F

## Ownership Suggestion

- Person 1: App customer experience
- Person 2: App dashboard and manager view
- Person 3: MLOps train/serve consistency
- Person 4: DataOps versioning and validation

## Dependencies

- Workstream A depends on prediction data already being available
- Workstream B may optionally consume monitoring outputs from Workstream F
- Workstream C should be completed before deeper registry and monitoring refinements
- Workstream D should be completed before finalizing run lineage in Workstream F
- Workstream E can begin immediately

## Definition of Done

A workstream is done only if:

- code is merged and documented
- relevant README instructions are updated
- the affected flow is manually verified or tested
- the UI or pipeline behavior is demonstrably improved

## Best Five If Time Is Limited

If the team can only finish five upgrades, prioritize:

1. App action recommendations
2. App manager dashboard
3. App business-value prioritization
4. MLOps full pipeline packaging for serving
5. Data and model version linkage across runs
