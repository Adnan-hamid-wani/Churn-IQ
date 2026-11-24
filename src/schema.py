"""The verified IBM input schema shared by training and inference."""

import numpy as np
import pandas as pd

ID_COLUMN = "customerID"
TARGET = "Churn"
DEMOGRAPHIC_COLUMNS = ("gender", "SeniorCitizen", "Partner", "Dependents")
NUMERIC_FEATURES = ("tenure", "MonthlyCharges", "TotalCharges")
YES_NO = ("No", "Yes")
INTERNET_ADDONS = (
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
)
CATEGORIES = {
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
    "PhoneService": YES_NO,
    "MultipleLines": ("No", "Yes", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    **dict.fromkeys(INTERNET_ADDONS, ("No", "Yes", "No internet service")),
}
FEATURES = (*NUMERIC_FEATURES, *CATEGORIES)
DATASET_COLUMNS = (ID_COLUMN, *DEMOGRAPHIC_COLUMNS, *FEATURES, TARGET)

FIELD_LABELS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple phone lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
}


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise a readable error for missing or duplicate headers."""
    duplicates = frame.columns[frame.columns.duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"Repeated columns: {', '.join(duplicates)}.")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")


def clean_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().replace("", pd.NA)


def _value_error(column: str, invalid: pd.Series, expectation: str) -> ValueError:
    rows = ", ".join(str(index + 1) for index in np.flatnonzero(invalid)[:5])
    return ValueError(f"{column}: {expectation} (data rows {rows}).")


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ordered model features, validating present values and preserving blanks."""
    require_columns(frame, FEATURES)
    result = pd.DataFrame(index=frame.index)
    for column in NUMERIC_FEATURES:
        values = clean_text(frame[column])
        numbers = pd.to_numeric(values, errors="coerce").astype("float64")
        invalid = values.notna() & (~np.isfinite(numbers) | (numbers < 0))
        if column == "tenure":
            invalid |= numbers.notna() & numbers.ne(np.floor(numbers))
        if invalid.any():
            expected = "use whole nonnegative months" if column == "tenure" else "use a finite nonnegative number"
            raise _value_error(column, invalid, expected)
        result[column] = numbers

    for column, options in CATEGORIES.items():
        values = clean_text(frame[column])
        invalid = values.notna() & ~values.isin(options)
        if invalid.any():
            raise _value_error(column, invalid, f"use one of {', '.join(options)}")
        result[column] = values.astype(object).where(values.notna(), np.nan)
    return result


def validate_service_combinations(frame: pd.DataFrame) -> None:
    """Reject contradictory supplied service fields while leaving blanks imputable."""
    pairs = [("MultipleLines", "PhoneService", "No phone service")]
    pairs.extend((column, "InternetService", "No internet service") for column in INTERNET_ADDONS)
    for child, parent, inactive in pairs:
        has_both = frame[child].notna() & frame[parent].notna()
        inconsistent = frame[child].eq(inactive).ne(frame[parent].eq("No"))
        invalid = has_both & inconsistent
        if invalid.any():
            raise _value_error(child, invalid, f"use a value consistent with {parent}")


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the IBM table and retain its non-model fields for analysis."""
    require_columns(frame, DATASET_COLUMNS)
    ids = clean_text(frame[ID_COLUMN])
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("customerID must contain unique, nonempty values.")
    labels = clean_text(frame[TARGET])
    if not labels.isin(YES_NO).all():
        raise ValueError("Churn must be Yes or No for every customer.")

    features = normalize_features(frame)
    validate_service_combinations(features)
    data = frame.loc[:, DATASET_COLUMNS].copy()
    for column in FEATURES:
        data[column] = features[column]
    data[ID_COLUMN] = ids
    data[TARGET] = labels
    data["SeniorCitizen"] = data["SeniorCitizen"].astype(int)
    return data
