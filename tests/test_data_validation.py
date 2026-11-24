import pandas as pd
import pytest

from src.schema import FEATURES, DEMOGRAPHIC_COLUMNS, normalize_features, validate_dataset


def sample_features():
    return {
        "tenure": "0",
        "MonthlyCharges": "35.60",
        "TotalCharges": " ",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    }


def test_features_ignore_target_and_demographics_and_keep_blanks():
    frame = pd.DataFrame([{**sample_features(), "gender": "Male", "Churn": "Yes"}])
    cleaned = normalize_features(frame)
    assert tuple(cleaned.columns) == FEATURES
    assert cleaned.loc[0, "MonthlyCharges"] == 35.60
    assert pd.isna(cleaned.loc[0, "TotalCharges"])


@pytest.mark.parametrize(
    ("column", "value"),
    [("tenure", "-1"), ("tenure", "2.5"), ("MonthlyCharges", "inf"),
     ("TotalCharges", "typo"), ("Contract", "monthly-ish")],
)
def test_invalid_feature_values_name_the_column(column, value):
    with pytest.raises(ValueError, match=column):
        normalize_features(pd.DataFrame([{**sample_features(), column: value}]))


def test_missing_headers_name_the_missing_column():
    row = sample_features()
    del row["Contract"]
    with pytest.raises(ValueError, match="Contract"):
        normalize_features(pd.DataFrame([row]))


def test_dataset_retains_demographics_and_rejects_duplicate_customer_id():
    demographics = dict(zip(DEMOGRAPHIC_COLUMNS, ["Female", "0", "No", "No"], strict=True))
    rows = [{**sample_features(), **demographics, "customerID": "001", "Churn": "Yes"}]
    assert validate_dataset(pd.DataFrame(rows)).loc[0, "gender"] == "Female"
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(pd.DataFrame(rows + rows))



def test_corrupt_local_copy_is_rejected_by_checksum(tmp_path):
    from src.data_loader import download_dataset

    sample = tmp_path / "sample.csv"
    sample.write_text("customerID,Churn", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        download_dataset(sample)
