# US-15 Model Governance Evidence

Use this checklist to verify the model-card acceptance criteria after running:

```bash
python source/mlops/train_model.py
```

## Expected local artifacts

Check `docs/artifacts/` for:

- `model_card.md`
- `feature_list_types.csv`
- `fairness_gender.csv`
- `fairness_age_band.csv`

## Expected MLflow evidence

Open the best-model run from `docs/artifacts/best_model.json` and confirm the run has:

- tag `model_governance=true`
- tag `production_candidate=true`
- artifacts:
  - `model_card.md`
  - `feature_list_types.csv`
  - `fairness_gender.csv`
  - `fairness_age_band.csv`

## Acceptance checks

- Model card contains training data description and date range
- Model card contains feature list and types
- Model card contains validation performance metrics
- Model card contains known limitations and bias considerations
- Fairness analysis includes churn rate by gender and age band
