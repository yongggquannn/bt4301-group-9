#!/bin/bash

mlflow server \
--backend-store-uri postgresql+psycopg2://postgres@localhost:5432/mlflow_db \
--default-artifact-root ./mlruns \
--host 0.0.0.0 \
--port 5001
