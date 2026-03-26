#!/bin/bash
set -e

# Load .env from project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Defaults: on macOS Homebrew, the superuser role matches the system username
MLFLOW_POSTGRES_USER="${MLFLOW_POSTGRES_USER:-$(whoami)}"
MLFLOW_POSTGRES_PASSWORD="${MLFLOW_POSTGRES_PASSWORD:-}"
MLFLOW_POSTGRES_DB="${MLFLOW_POSTGRES_DB:-mlflow_db}"
MLFLOW_PORT="${MLFLOW_PORT:-5001}"

# Build connection URI (omit password when using peer/trust auth on macOS)
if [ -n "$MLFLOW_POSTGRES_PASSWORD" ]; then
  DB_URI="postgresql+psycopg2://${MLFLOW_POSTGRES_USER}:${MLFLOW_POSTGRES_PASSWORD}@localhost:5432/${MLFLOW_POSTGRES_DB}"
else
  DB_URI="postgresql+psycopg2://${MLFLOW_POSTGRES_USER}@localhost:5432/${MLFLOW_POSTGRES_DB}"
fi

# Create the MLflow database if it doesn't exist
psql -U "$MLFLOW_POSTGRES_USER" -tc \
  "SELECT 1 FROM pg_database WHERE datname='${MLFLOW_POSTGRES_DB}'" \
  | grep -q 1 \
  || psql -U "$MLFLOW_POSTGRES_USER" -c "CREATE DATABASE ${MLFLOW_POSTGRES_DB}"

mlflow server \
  --backend-store-uri "$DB_URI" \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT"
