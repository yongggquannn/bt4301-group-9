# US-20 Champion-Challenger Evidence Guide

## Goal
Show that a newly registered model version is only promoted to Production when it beats the current champion by more than the configured AUC threshold.

## Run the registry script
```bash
python source/mlops/register_model.py --promotion-threshold 0.005
```

## Check evidence artifacts
Confirm these files exist:

- `docs/artifacts/model_registry.json`
- `docs/artifacts/champion_challenger_registry.json`

The champion-challenger JSON should include:

- `challenger_auc`
- `champion_version`
- `champion_auc`
- `promotion_threshold`
- `auc_margin`
- `decision`
- `final_stage`
- `transitions`

## Check MLflow Model Registry UI
Open:

- `http://localhost:5001/#/models/KKBox-Churn-Classifier`

Capture a screenshot showing:

- version history across Sprint 2 and Sprint 3
- the current Production champion
- the newly registered challenger version
- model version tags or description containing the promotion decision

## Expected behavior
- If `new_model_auc > champion_auc + threshold`, the challenger becomes Production.
- Otherwise, the existing champion stays in Production and the new version remains in Staging as a challenger.
