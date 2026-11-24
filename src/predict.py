"""Score customer profiles and parse or export batch CSV files."""

import csv
from dataclasses import dataclass
from io import StringIO

import pandas as pd

from src.evaluate import churn_probabilities, classify
from src.schema import (
    FEATURES, ID_COLUMN, clean_text, normalize_features, require_columns,
    validate_service_combinations,
)

MAX_CSV_BYTES = 2_000_000
MAX_CSV_ROWS = 10_000


@dataclass(frozen=True)
class Prediction:
    probability: float
    predicted_class: str
    risk: str


def risk_band(probability: float, bundle: dict) -> str:
    if probability >= bundle["threshold"]:
        return "High"
    if probability >= bundle["risk_low_cutoff"]:
        return "Medium"
    return "Low"


def _validated_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = normalize_features(frame)
    if features.empty or features.isna().all(axis=1).any():
        raise ValueError("Every customer row must have at least one model input value.")
    validate_service_combinations(features)
    return features


def predict_one(bundle: dict, profile: dict) -> Prediction:
    features = _validated_features(pd.DataFrame([profile]))
    probabilities = churn_probabilities(bundle["pipeline"], features)
    probability = float(probabilities[0])
    klass = "Yes" if classify(probabilities, bundle["threshold"])[0] else "No"
    return Prediction(probability, klass, risk_band(probability, bundle))


def predict_batch(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Return just the ID, predicted class, and probability in source row order."""
    require_columns(frame, (ID_COLUMN, *FEATURES))
    ids = clean_text(frame[ID_COLUMN])
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("customerID must contain unique, nonempty values.")

    features = _validated_features(frame)
    probabilities = churn_probabilities(bundle["pipeline"], features)
    classes = classify(probabilities, bundle["threshold"])
    return pd.DataFrame({
        ID_COLUMN: ids.tolist(),
        "predicted_class": ["Yes" if label else "No" for label in classes],
        "churn_probability": probabilities,
    })


def parse_batch(content: bytes) -> pd.DataFrame:
    """Preserve input text, including leading zeros, and reject malformed CSV rows."""
    if len(content) > MAX_CSV_BYTES:
        raise ValueError("Upload a CSV no larger than 2 MB.")
    try:
        reader = csv.reader(StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        header = [column.strip() for column in next(reader, [])]
        if not header:
            raise ValueError("The CSV needs a header row.")
        require_columns(pd.DataFrame(columns=header), ())
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV line {reader.line_num} needs {len(header)} fields.")
            rows.append(row)
            if len(rows) > MAX_CSV_ROWS:
                raise ValueError(f"Upload at most {MAX_CSV_ROWS:,} customer rows.")
    except UnicodeDecodeError as error:
        raise ValueError("Save the CSV as UTF-8 and try again.") from error
    except csv.Error as error:
        raise ValueError("The CSV could not be read. Check its quoting and comma separators.") from error
    if not rows:
        raise ValueError("The CSV needs at least one customer row.")
    return pd.DataFrame(rows, columns=header)


def export_csv(result: pd.DataFrame) -> bytes:
    output = result.copy()
    # A leading apostrophe prevents spreadsheet apps from interpreting the ID as a formula.
    output[ID_COLUMN] = output[ID_COLUMN].map(
        lambda value: "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value
    )
    return output.to_csv(index=False, lineterminator="\n").encode("utf-8")


def example_csv() -> bytes:
    """Return a pair of fictional customer profiles used to demonstrate the interface."""
    rows = [
        ["DEMO-001", 4, 81.40, 324.50, "Month-to-month", "Yes", "Electronic check",
         "Yes", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "No"],
        ["DEMO-002", 46, 62.10, 2792.00, "Two year", "No", "Bank transfer (automatic)",
         "Yes", "Yes", "DSL", "Yes", "Yes", "No", "Yes", "No", "No"],
    ]
    return pd.DataFrame(rows, columns=(ID_COLUMN, *FEATURES)).to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
