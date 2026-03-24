# MLflow Setup (US-09)

## MLflow Tracking Server

The MLflow tracking server is configured with:

Backend store: PostgreSQL
Artifact store: local directory (mlruns)

### Start MLflow Server

mlflow server \
--backend-store-uri postgresql+psycopg2://postgres@localhost:5432/mlflow_db \
--default-artifact-root ./mlruns \
--host 0.0.0.0 \
--port 5001

### Access the UI

Open in browser:

http://localhost:5001

### Verification

The MLflow UI loads successfully and shows the default experiment.
