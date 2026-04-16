from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature  # still used by test_airflow_load_production_model_writes_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.mlops import register_model, score_churn

pytestmark = pytest.mark.mlops


def test_extract_input_columns_from_signature() -> None:
    features = pd.DataFrame(
        {
            "num_feature": [1.0, 2.0],
            "cat_feature": ["a", "b"],
        }
    )
    signature = infer_signature(features, np.array([[0.9, 0.1], [0.2, 0.8]]))

    assert score_churn._extract_input_columns(signature) == [
        "num_feature",
        "cat_feature",
    ]


def test_prepare_model_inputs_orders_and_filters_columns() -> None:
    df = pd.DataFrame(
        {
            "extra_feature": [99, 100],
            "cat_feature": ["a", "b"],
            "num_feature": [1.0, 2.0],
        }
    )

    prepared = score_churn._prepare_model_inputs(df, ["num_feature", "cat_feature"])

    assert list(prepared.columns) == ["num_feature", "cat_feature"]
    assert prepared.to_dict(orient="list") == {
        "num_feature": [1.0, 2.0],
        "cat_feature": ["a", "b"],
    }


def test_prepare_model_inputs_fails_on_missing_columns() -> None:
    df = pd.DataFrame({"num_feature": [1.0, 2.0]})

    with pytest.raises(RuntimeError, match="missing required production-model columns"):
        score_churn._prepare_model_inputs(df, ["num_feature", "cat_feature"])


def test_airflow_load_production_model_writes_metadata(tmp_path, monkeypatch) -> None:
    features = pd.DataFrame(
        {
            "num_feature": [1.0, 2.0],
            "cat_feature": ["a", "b"],
        }
    )
    signature = infer_signature(features, np.array([[0.9, 0.1], [0.2, 0.8]]))
    model_info = SimpleNamespace(
        signature=signature,
        name="KKBox-Churn-Classifier",
        registered_model_version="7",
    )

    monkeypatch.setattr(score_churn, "_resolve_model_uri", lambda: "models:/KKBox-Churn-Classifier/Production")
    monkeypatch.setattr(score_churn, "_load_model_info", lambda uri: model_info)
    monkeypatch.setattr(score_churn, "_load_serving_model", lambda uri: object())

    artifacts = score_churn.ScoringArtifacts(
        features_path=tmp_path / "features.parquet",
        model_metadata_path=tmp_path / "production_model.json",
        scores_path=tmp_path / "scores.parquet",
    )

    score_churn.airflow_load_production_model(artifacts)
    metadata = score_churn._load_model_metadata(artifacts.model_metadata_path)

    assert metadata.model_uri == "models:/KKBox-Churn-Classifier/Production"
    assert metadata.input_columns == ["num_feature", "cat_feature"]
    assert metadata.model_name == "KKBox-Churn-Classifier"
    assert metadata.model_version == "7"


def test_validate_model_artifact_rejects_wrong_artifact_type() -> None:
    best_info = {"model_artifact_type": "raw_estimator", "input_features": ["f1"]}

    with pytest.raises(RuntimeError, match="not a serving_pipeline"):
        register_model.validate_model_artifact(best_info, "runs:/abc123/model")


def test_validate_model_artifact_rejects_missing_input_features() -> None:
    best_info = {"model_artifact_type": "serving_pipeline", "input_features": []}

    with pytest.raises(RuntimeError, match="no input_features"):
        register_model.validate_model_artifact(best_info, "runs:/abc123/model")


def test_validate_model_artifact_accepts_valid_best_info() -> None:
    best_info = {
        "model_artifact_type": "serving_pipeline",
        "input_features": ["num_feature", "cat_feature"],
    }

    register_model.validate_model_artifact(best_info, "runs:/abc123/model")
