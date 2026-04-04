# US-24 GitHub Actions CI/CD Evidence Guide

## Goal
Show that the repository has a GitHub Actions workflow that:

- triggers on push to `main`
- triggers on a model promotion event from MLflow
- runs `lint -> test -> build Docker image -> deploy`

## Workflow file
Confirm this file exists:

- `.github/workflows/deploy.yml`

## Trigger sources
The workflow should start on:

- `push` to `main`
- `repository_dispatch` with event type `mlflow-model-promoted`

## Model-promotion trigger
When `source/mlops/register_model.py` promotes a challenger to `Production`, it can emit a GitHub repository dispatch if these environment variables are set:

- `GITHUB_DISPATCH_REPO`
- `GITHUB_DISPATCH_TOKEN`
- optional: `GITHUB_DISPATCH_EVENT_TYPE`

## Expected pipeline stages
Successful run should show these jobs in order:

- `lint`
- `test`
- `build`
- `deploy`

## Screenshot
Capture a screenshot from GitHub Actions showing:

- workflow name `Deploy Web App`
- successful run on `main` push or `mlflow-model-promoted`
- successful `lint`, `test`, `build`, and `deploy` jobs
