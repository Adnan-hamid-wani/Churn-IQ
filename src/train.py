"""Train, compare and export the ChurnIQ model on real IBM data."""

import argparse
import json
import platform
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from src.artifacts import FORMAT_VERSION, METRICS_FILE, MODEL_DIR, PIPELINE_FILE
from src.data_loader import DATA_SHA256, load_dataset
from src.evaluate import churn_probabilities, evaluate_scores, evaluation_curves, select_threshold
from src.explain import build_references, global_importance
from src.preprocessing import MODEL_GRIDS, SEED, build_pipeline
from src.schema import FEATURES, TARGET

def split_dataset(features: pd.DataFrame, labels: pd.Series) -> tuple:
    """Create disjoint stratified train, validation and held-out test partitions."""
    development, test, dev_labels, test_labels = train_test_split(
        features, labels, test_size=0.20, stratify=labels, random_state=SEED,
    )
    training, validation, train_labels, validation_labels = train_test_split(
        development, dev_labels, test_size=0.25, stratify=dev_labels, random_state=SEED,
    )
    return (training, train_labels), (validation, validation_labels), (test, test_labels)


def write_artifacts(bundle: dict, metrics: dict, directory: Path) -> None:
    """Stage both files before replacing the released artifact pair."""
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as temporary:
        staged = Path(temporary)
        pipeline_path = staged / PIPELINE_FILE
        metrics_path = staged / METRICS_FILE
        joblib.dump(bundle, pipeline_path, compress=3)
        metrics["pipeline_sha256"] = sha256(pipeline_path.read_bytes()).hexdigest()
        metrics_path.write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        pipeline_path.replace(directory / PIPELINE_FILE)
        metrics_path.replace(directory / METRICS_FILE)


def train(directory: Path = MODEL_DIR) -> dict:
    data = load_dataset()
    features = data.loc[:, FEATURES]
    labels = data[TARGET].eq("Yes").astype(int)
    (
        (training, train_labels),
        (validation, validation_labels),
        (test, test_labels),
    ) = split_dataset(features, labels)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    searches = {}
    comparison = []
    for name, grid in MODEL_GRIDS.items():
        search = GridSearchCV(
            build_pipeline(name), grid, scoring="average_precision", cv=folds, error_score="raise",
        ).fit(training, train_labels)
        searches[name] = search
        comparison.append({
            "model": name,
            "cv_average_precision": float(search.best_score_),
            "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "parameters": search.best_params_,
        })
        print(f"{name}: CV average precision {search.best_score_:.4f}", flush=True)

    comparison.sort(key=lambda row: row["cv_average_precision"], reverse=True)
    selected = comparison[0]["model"]
    pipeline = searches[selected].best_estimator_
    validation_probabilities = churn_probabilities(pipeline, validation)
    threshold = select_threshold(validation_labels, validation_probabilities)
    test_probabilities = churn_probabilities(pipeline, test)
    bundle = {
        "format_version": FORMAT_VERSION,
        "model_name": selected,
        "pipeline": pipeline,
        "threshold": threshold,
        "risk_low_cutoff": threshold / 2,
        "references": build_references(training),
    }
    metrics = {
        "format_version": FORMAT_VERSION,
        "dataset_sha256": DATA_SHA256,
        "seed": SEED,
        "features": list(FEATURES),
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "splits": {
            name: {"rows": len(split), "positive": int(split.sum())}
            for name, split in (
                ("train", train_labels), ("validation", validation_labels), ("test", test_labels),
            )
        },
        "cv_folds": folds.n_splits,
        "selection_metric": "average_precision",
        "comparison": comparison,
        "selected_model": selected,
        "threshold": threshold,
        "risk_low_cutoff": bundle["risk_low_cutoff"],
        "validation": {
            "rule": "Maximize positive-class F1 on validation; prefer higher recall on ties.",
            "chosen": evaluate_scores(validation_labels, validation_probabilities, threshold),
            "at_0_50": evaluate_scores(validation_labels, validation_probabilities, 0.5),
        },
        "test": evaluate_scores(test_labels, test_probabilities, threshold),
        "test_curves": evaluation_curves(test_labels, test_probabilities),
        "global_importance": global_importance(pipeline, test, test_labels),
    }
    write_artifacts(bundle, metrics, directory)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        result = train(arguments.output)
    print(f"Selected {result['selected_model']} at threshold {result['threshold']:.6f}.")
    print(json.dumps(result["test"], indent=2))
