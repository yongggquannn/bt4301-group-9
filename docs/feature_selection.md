# Feature engineering & selection (US: Feature Selection)

This project builds a feature store table at `processed.customer_features` (see `source/dataops/build_customer_features.py`).  
This document describes how the **final feature set** is selected to ensure features are predictive and non-redundant.

## Inputs

- **Feature store**: `processed.customer_features`
- **Target**: `is_churn`
- **Identifier** (excluded): `msno`
- **Excluded system column** (excluded): `feature_created_at`

## Importance analysis completed

We run both:

- **Permutation importance** on a validation split using **ROC AUC** as the scoring metric
- **Model-based feature importance** (Random Forest impurity importance) on the transformed feature space

- **Estimator**: `RandomForestClassifier` in a preprocessing pipeline
- **Preprocessing**:
  - Numeric: median imputation
  - Categorical (`gender`, and key coded columns if present): most-frequent imputation + one-hot encoding

Permutation importance is computed at the **original-column level** by permuting raw columns in \(X_{val}\) and measuring the drop in ROC AUC.

## Final feature set (count + selection method)

Selection method (implemented in `source/mlops/feature_selection.py`):

1. Rank features by permutation importance mean.
2. Keep features with `perm_importance_mean > 0.0` (configurable), ensuring at least `min_features` are kept.
3. Reduce redundancy among **numeric** features by dropping features with absolute correlation \(\ge 0.9\) (configurable), keeping the more important feature.

The resulting **final feature set count** and **feature list** are produced by the script and saved as artifacts.

## MLflow logging (artifact)

The feature list is logged to MLflow as:

- `docs/artifacts/final_feature_set.json`
- `docs/artifacts/final_feature_set.csv`
- `docs/artifacts/permutation_importance.csv`
- `docs/artifacts/model_feature_importance.csv`

Run:

```bash
python source/mlops/feature_selection.py
```

