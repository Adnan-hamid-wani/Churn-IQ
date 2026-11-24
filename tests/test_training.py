import numpy as np
import pandas as pd

from src.evaluate import evaluate_scores, select_threshold
from src.preprocessing import build_pipeline
from src.schema import CATEGORIES, FEATURES, NUMERIC_FEATURES
from src.train import split_dataset


def test_pipeline_handles_missing_and_unknown_categories():
    row = {feature: 10.0 for feature in NUMERIC_FEATURES}
    row.update({feature: options[0] for feature, options in CATEGORIES.items()})
    first = {**row, "tenure": 1.0}
    last = {**row, "tenure": 50.0}
    training = pd.DataFrame([first, first, last, last], columns=FEATURES)
    training.loc[0, "TotalCharges"] = np.nan
    training.loc[1, "PaymentMethod"] = np.nan
    pipeline = build_pipeline("Logistic regression").fit(training, [1, 1, 0, 0])
    future = pd.DataFrame([{**first, "Contract": "Future plan", "tenure": np.nan}])
    probabilities = pipeline.predict_proba(future)
    assert probabilities.shape == (1, 2)
    assert ((0 <= probabilities) & (probabilities <= 1)).all()
    assert np.isclose(probabilities.sum(), 1)


def test_threshold_is_selected_from_validation_and_applied_inclusively():
    labels = pd.Series([0, 1, 0, 1, 0, 1])
    probabilities = np.array([0.10, 0.49, 0.20, 0.80, 0.43, 0.48])
    threshold = select_threshold(labels, probabilities)
    assert threshold == 0.48
    assert evaluate_scores(labels, probabilities, threshold)["confusion_matrix"] == [[3, 0], [0, 3]]


def test_split_is_stratified_disjoint_and_repeatable():
    features = pd.DataFrame({"x": range(100)})
    target = pd.Series([0, 0, 0, 1] * 25)
    first = split_dataset(features, target)
    second = split_dataset(features, target)
    assert [len(labels) for _, labels in first] == [60, 20, 20]
    assert [labels.mean() for _, labels in first] == [0.25] * 3
    assert sorted(index for frame, _ in first for index in frame.index) == list(range(100))
    assert [frame.index.tolist() for frame, _ in first] == [frame.index.tolist() for frame, _ in second]
