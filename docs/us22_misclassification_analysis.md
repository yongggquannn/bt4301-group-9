# US-22 Misclassification Analysis

## Validation confusion matrix breakdown

- Model: `xgboost`
- Decision threshold: `0.5`
- TN: 34511, FP: 835, FN: 535, TP: 2958
- Missed churn (FN): **535**
- False alarm (FP): **835**

## Feature distribution comparison (misclassified vs correctly classified)

- Numeric summary: `docs/artifacts/us22_numeric_distribution_comparison.csv`
- Categorical summary: `docs/artifacts/us22_categorical_distribution_comparison.csv`
- Plot: `docs/artifacts/us22_feature_distribution_plot.png`

## Actionable insights

1. `transaction_count` is materially higher among misclassified cases (misclassified mean=2.16, correct mean=1.18). Action: add bucketized/interaction features for `transaction_count` and re-tune threshold.
2. `total_amount_paid` is materially higher among misclassified cases (misclassified mean=320.97, correct mean=171.42). Action: add bucketized/interaction features for `total_amount_paid` and re-tune threshold.
3. Category `latest_is_auto_renew=1.0` is over-represented in misclassifications (misclassified share=49.3%, correct share=89.0%). Action: introduce category-target interactions or category-specific calibration.
4. False positives are at least as frequent as false negatives. Action: increase decision threshold or add precision-oriented features to reduce false alarms.
