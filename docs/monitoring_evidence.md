# US-14 Monitoring Evidence

Use this checklist to capture evidence for the weekly monitoring user story.

## 1) Airflow DAG screenshot

Capture a screenshot of DAG `weekly_model_monitoring` in Airflow Graph/Grid view after a run.

## 2) Sample monitoring results

Run the SQL below and screenshot the output table:

```sql
SELECT
  monitored_at,
  baseline_auc_source,
  cohort_strategy,
  baseline_auc,
  current_auc,
  auc_delta,
  max_psi,
  breached,
  breached_reasons,
  baseline_row_count,
  current_row_count,
  psi_by_feature
FROM processed.model_monitoring_results
ORDER BY monitored_at DESC
LIMIT 10;
```

Acceptance checks:

- PSI is logged for top 5 features in `psi_by_feature`
- AUC baseline/current and `auc_delta` are logged
- `baseline_auc_source` shows whether the degradation baseline came from the production model artifact or historical window fallback
- `breached = true` when `max_psi > 0.2` or `auc_delta > 0.05`
