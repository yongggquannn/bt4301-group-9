-- US-19: Add SHAP explainability column to churn predictions.
-- Run this migration on existing databases where the table already exists.
ALTER TABLE processed.churn_predictions
  ADD COLUMN IF NOT EXISTS shap_values JSONB;
