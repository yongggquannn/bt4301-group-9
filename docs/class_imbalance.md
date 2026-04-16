# Class Imbalance Handling

## Comparison

Saved to: `docs/artifacts/precision_recall_f1_comparison.csv`

| Strategy | Precision (churn) | Recall (churn) | F1 (churn) | ROC AUC |
|---|---:|---:|---:|---:|
| SMOTE | 0.5066 | 0.5308 | 0.5184 | 0.8668 |
| class_weight="balanced" | 0.3314 | 0.7043 | 0.4507 | 0.8662 |

## Chosen strategy

Chosen: **smote**

Rule: choose max churn F1; tie-break with churn recall.

Justification:
```json
{
  "chosen_strategy": "smote",
  "rule": "Choose max churn F1; tie-break with churn recall.",
  "metrics": {
    "smote": {
      "precision_churn": 0.5065573770491804,
      "recall_churn": 0.5307758373890639,
      "f1_churn": 0.5183838948692856,
      "roc_auc": 0.8668130734069606,
      "support_churn": 3493
    },
    "class_weight_balanced": {
      "precision_churn": 0.3314023979523104,
      "recall_churn": 0.704265674205554,
      "f1_churn": 0.4507145474532796,
      "roc_auc": 0.8662145487149255,
      "support_churn": 3493
    }
  }
}
```
