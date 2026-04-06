# US-22 Misclassification Analysis

## Validation confusion matrix breakdown

- Model: `xgboost`
- Decision threshold: `0.5`
- TN: 34545, FP: 801, FN: 511, TP: 2982
- Missed churn (FN): **511**
- False alarm (FP): **801**

## Feature distribution comparison (misclassified vs correctly classified)

- Numeric summary: `docs/artifacts/numeric_distribution_comparison.csv`
- Categorical summary: `docs/artifacts/categorical_distribution_comparison.csv`
- Plot: `docs/artifacts/feature_distribution_plot.png`

## Actionable insights

1. `total_amount_paid` is materially higher among misclassified cases (misclassified mean=357.10, correct mean=172.01). Action: add bucketized/interaction features for `total_amount_paid` and re-tune threshold.
2. `transaction_count` is materially higher among misclassified cases (misclassified mean=2.41, correct mean=1.18). Action: add bucketized/interaction features for `transaction_count` and re-tune threshold.
3. Category `latest_is_auto_renew=1.0` is over-represented in misclassifications (misclassified share=47.9%, correct share=88.9%). Action: introduce category-target interactions or category-specific calibration.
4. False positives are at least as frequent as false negatives. Action: increase decision threshold or add precision-oriented features to reduce false alarms.
