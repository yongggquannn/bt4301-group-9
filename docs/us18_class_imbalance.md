# US18 - Class Imbalance Handling

## Comparison

Saved to: `docs/artifacts/us18_precision_recall_f1_comparison.csv`

| Strategy | Precision (churn) | Recall (churn) | F1 (churn) | ROC AUC |
|---|---:|---:|---:|---:|
| SMOTE | 0.5135 | 0.5385 | 0.5257 | 0.8684 |
| class_weight="balanced" | 0.3312 | 0.7106 | 0.4518 | 0.8671 |

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
      "precision_churn": 0.5135135135135135,
      "recall_churn": 0.5385055825937589,
      "f1_churn": 0.5257126886528787,
      "roc_auc": 0.8683583469450399,
      "support_churn": 3493
    },
    "class_weight_balanced": {
      "precision_churn": 0.33124249299346054,
      "recall_churn": 0.7105639851130833,
      "f1_churn": 0.4518478062989259,
      "roc_auc": 0.8670593606156465,
      "support_churn": 3493
    }
  }
}
```
