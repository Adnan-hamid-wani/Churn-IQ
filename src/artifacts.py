"""Load the trusted model and verify that it matches the released metrics."""

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import joblib
import sklearn

from src.data_loader import DATA_SHA256, ROOT
from src.schema import FEATURES

MODEL_DIR = ROOT / "models"
PIPELINE_FILE = "churn_pipeline.joblib"
METRICS_FILE = "metrics.json"
FORMAT_VERSION = 1


def load_release(directory: Path = MODEL_DIR) -> tuple[dict, dict]:
    """Open the project-owned artifact pair. Only use paths trusted by the application."""
    metrics = json.loads((directory / METRICS_FILE).read_text(encoding="utf-8"))
    pipeline_bytes = (directory / PIPELINE_FILE).read_bytes()
    if sha256(pipeline_bytes).hexdigest() != metrics.get("pipeline_sha256"):
        raise ValueError("Model checksum does not match the accompanying metrics.")
    if (
        metrics.get("format_version") != FORMAT_VERSION
        or tuple(metrics.get("features", ())) != FEATURES
        or metrics.get("dataset_sha256") != DATA_SHA256
    ):
        raise ValueError("Model artifacts do not match the current ChurnIQ schema.")
    if metrics["versions"]["scikit-learn"] != sklearn.__version__:
        raise ValueError("Install the pinned scikit-learn version before loading the model.")

    bundle = joblib.load(BytesIO(pipeline_bytes))
    if (
        bundle["format_version"] != FORMAT_VERSION
        or bundle["model_name"] != metrics["selected_model"]
        or tuple(bundle["pipeline"].feature_names_in_) != FEATURES
        or bundle["threshold"] != metrics["threshold"]
        or not (0 <= bundle["risk_low_cutoff"] == metrics["risk_low_cutoff"] <= bundle["threshold"] <= 1)
    ):
        raise ValueError("The model bundle and evaluation metadata are inconsistent.")
    return bundle, metrics
