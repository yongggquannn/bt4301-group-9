-- Raw schema
CREATE SCHEMA IF NOT EXISTS raw;
CREATE TABLE IF NOT EXISTS raw.members (...);
CREATE TABLE IF NOT EXISTS raw.transactions (...);
CREATE TABLE IF NOT EXISTS raw.user_logs (...);
CREATE TABLE IF NOT EXISTS raw.train_labels (...);

-- Processed schema
CREATE SCHEMA IF NOT EXISTS processed;
CREATE TABLE IF NOT EXISTS processed.features (...);

-- Predictions schema
CREATE SCHEMA IF NOT EXISTS predictions;
CREATE TABLE IF NOT EXISTS predictions.churn_predictions (...);

-- Lineage schema
CREATE SCHEMA IF NOT EXISTS lineage;
CREATE TABLE IF NOT EXISTS lineage.data_lineage (...);