# US-21 Automated Retraining DAG Evidence Guide

## Goal
Show that the retraining DAG only runs the full retraining path when the latest monitoring result breaches the drift/degradation thresholds from US-14.

## Trigger the DAG
Open Airflow and trigger:

- `automated_retraining`

## DAG structure
Expected task chain:

- `check_drift_results -> retrain_if_needed -> evaluate -> register`

There is also a conditional skip path:

- `check_drift_results -> skip_retraining`

## Trigger condition
The DAG should retrain only when the latest row in `processed.model_monitoring_results` satisfies at least one:

- `max_psi > 0.2`
- `auc_delta > 0.05`

## Check evidence artifacts
Confirm these files exist after a triggered retraining run:

- `docs/artifacts/retraining_evaluation.json`
- `docs/artifacts/retraining_decision.json`

`retraining_evaluation.json` should include the newly selected best model and metrics from the retraining run.

`retraining_decision.json` should include:

- the registration timestamp
- the promotion threshold passed to champion-challenger registration
- the evaluation payload
- the final registry decision from `champion_challenger_registry.json`

## Airflow screenshot
Capture a screenshot showing:

- the DAG Graph or Grid view
- the conditional branch from `check_drift_results`
- either the retraining path or the skip path
