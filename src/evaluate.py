"""Validation threshold selection and binary classification evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def churn_probabilities(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Get the probability for class 1 independently of the class array order."""
    class_index = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(features)[:, class_index]


def classify(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def select_threshold(labels: pd.Series, probabilities: np.ndarray) -> float:
    """Maximize validation F1, preferring more recall and then a higher cutoff for ties."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    totals = precision + recall
    f1 = np.divide(2 * precision * recall, totals, out=np.zeros_like(totals), where=totals > 0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    index = max(best, key=lambda i: (recall[i], thresholds[i]))
    return float(thresholds[index])


def evaluate_scores(labels: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = classify(probabilities, threshold)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def evaluation_curves(labels: pd.Series, probabilities: np.ndarray) -> dict:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return {
        "roc": {"fpr": false_positive_rate.tolist(), "tpr": true_positive_rate.tolist()},
        "precision_recall": {"recall": recall.tolist(), "precision": precision.tolist()},
        "prevalence": float(labels.mean()),
    }
